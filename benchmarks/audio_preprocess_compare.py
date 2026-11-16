"""Measure Omni's CPU audio input path without loading weights or using the GPU.

Run inside the serving image. Fixtures and processor initialization are outside
timing. No requests are sent to the running servers. Serialize this benchmark
with serving measurements: its CPU thread sweeps can otherwise perturb them.
"""

import argparse
import base64
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import statistics
import time
import wave

import av
import numpy as np
import torch
from transformers import AutoProcessor

from vllm.multimodal.audio import AudioResampler, resample_audio_pyav
from vllm.multimodal.media.audio import AudioMediaIO


def wav_bytes(waveform, sample_rate):
    pcm = np.rint(np.clip(waveform, -1, 32767 / 32768) * 32768).astype("<i2")
    output = BytesIO()
    with wave.open(output, "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(sample_rate)
        clip.writeframes(pcm.tobytes())
    return output.getvalue()


def opus_bytes(waveform, sample_rate=48000):
    """Match the browser recording's codec/rate, excluding encoding from timing."""
    output = BytesIO()
    with av.open(output, mode="w", format="webm") as container:
        stream = container.add_stream("libopus", rate=sample_rate)
        stream.layout = "mono"
        stream.bit_rate = 128000
        for offset in range(0, len(waveform), 960):
            chunk = np.ascontiguousarray(waveform[offset : offset + 960], dtype=np.float32)
            frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="flt", layout="mono")
            frame.sample_rate = sample_rate
            frame.pts = offset
            frame.time_base = Fraction(1, sample_rate)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return output.getvalue()


def fixtures(source, duration):
    samples = round(duration * 16000)
    signal = np.tile(source, (samples + len(source) - 1) // len(source))[:samples]
    high_rate = resample_audio_pyav(signal, orig_sr=16000, target_sr=48000)
    return (
        ("wav_16k", "audio/wav", wav_bytes(signal, 16000)),
        ("wav_48k", "audio/wav", wav_bytes(high_rate, 48000)),
        ("opus_webm_48k", "audio/webm", opus_bytes(high_rate)),
    )


def pad_to_hop_length(waveform, hop_length):
    # Match Qwen3OmniMoeThinkerMultiModalProcessor._get_hf_mm_data before HF.
    remainder = waveform.shape[-1] % hop_length
    return np.pad(waveform, (0, hop_length - remainder)) if remainder else waveform


def measure(call, warmups, repeats):
    for _ in range(warmups):
        call()
    timings = []
    for _ in range(repeats):
        began = time.perf_counter()
        result = call()
        timings.append((time.perf_counter() - began) * 1000)
    return {"median_ms": statistics.median(timings), "samples_ms": timings}, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Cached Omni snapshot; AutoProcessor only, no model weights")
    parser.add_argument("--audio", type=Path, required=True,
                        help="Speech fixture; repeated/cropped to the requested durations")
    parser.add_argument("--threads", type=int, nargs="+", default=(1, 2, 4, 8, 32))
    parser.add_argument("--durations", type=float, nargs="+", default=(2, 10, 30))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-fixtures", type=Path,
                        help="Optionally retain generated WAV/WebM files for serving comparisons")
    args = parser.parse_args()
    if min(args.threads) < 1 or min(args.durations) <= 0 or max(args.durations) > 30:
        parser.error("Thread counts must be positive; durations must be in (0,30] seconds")
    if args.warmups < 1 or args.repeats < 3:
        parser.error("Use at least one warmup and three measured repetitions")

    torch.set_num_threads(1)
    media = AudioMediaIO()
    source, rate = media.load_bytes(args.audio.read_bytes())
    source = resample_audio_pyav(source, orig_sr=rate, target_sr=16000)
    if not len(source):
        parser.error("The speech fixture is empty")
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True, use_fast=True)
    extractor = processor.feature_extractor
    if extractor.sampling_rate != 16000:
        parser.error("This benchmark expects Omni's 16000Hz feature extractor")
    if getattr(extractor, "dither", 0):
        parser.error("Cross-thread feature checks require deterministic extraction (dither=0)")
    resampler = AudioResampler(target_sr=extractor.sampling_rate)

    report = {
        "at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "torch": torch.__version__,
        "processor": type(processor).__name__, "feature_extractor": type(extractor).__name__,
        "checkpoint": str(args.checkpoint), "warmups": args.warmups, "repeats": args.repeats,
        "interop_threads": torch.get_num_interop_threads(),
        "environment": {name: os.environ.get(name) for name in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "scope": "CPU decode/resample/HF processing only; no model, GPU, HTTP, or microphone capture timing",
        "timing_note": "Stages are measured separately; their medians should not be added together",
        "cases": [],
    }

    def save():
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")

    def hf_process(waveform):
        return processor(
            text="<|audio_start|><|audio_pad|><|audio_end|>",
            audio=[pad_to_hop_length(waveform, extractor.hop_length)],
            audio_kwargs={}, text_kwargs={}, return_tensors="pt",
        )

    # Prepare codec variants before timing or changing thread counts. Distinct
    # formats are not expected to have identical features: Opus is lossy.
    for duration in args.durations:
        for name, mime, data in fixtures(source, duration):
            if args.save_fixtures:
                args.save_fixtures.mkdir(parents=True, exist_ok=True)
                suffix = "webm" if "webm" in mime else "wav"
                (args.save_fixtures / f"{name}-{duration:g}s.{suffix}").write_bytes(data)
            encoded = base64.b64encode(data).decode("ascii")
            decoded, sample_rate = media.load_bytes(data)
            normalized = resampler.resample(decoded, orig_sr=sample_rate)
            torch.set_num_threads(1)
            reference = hf_process(normalized)["input_features"].clone()

            def pipeline():
                waveform, rate = media.load_base64(mime, encoded)
                return hf_process(resampler.resample(waveform, orig_sr=rate))

            for threads in args.threads:
                torch.set_num_threads(threads)
                stages = {}
                for label, call in (
                    ("decode", lambda: media.load_bytes(data)),
                    ("base64_and_decode", lambda: media.load_base64(mime, encoded)),
                    ("resample", lambda: resampler.resample(decoded, orig_sr=sample_rate)),
                    ("hf_processor", lambda: hf_process(normalized)),
                    ("whole_cpu_pipeline", pipeline),
                ):
                    stages[label], result = measure(call, args.warmups, args.repeats)
                actual = result["input_features"]
                if actual.device.type != "cpu" or not torch.isfinite(actual).all():
                    raise RuntimeError("Expected finite CPU audio features")
                torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
                row = {
                    "format": name, "requested_duration_s": duration,
                    "decoded_duration_s": len(decoded) / sample_rate,
                    "encoded_bytes": len(data), "decoded_sample_rate": sample_rate,
                    "fixture_sha256": hashlib.sha256(data).hexdigest(),
                    "torch_threads": torch.get_num_threads(), "feature_shape": list(actual.shape),
                    "max_abs_feature_difference_vs_one_thread": float((actual - reference).abs().max()),
                    "stages": stages,
                }
                report["cases"].append(row)
                save()
                print(json.dumps({key: value for key, value in row.items() if key != "stages"}
                                 | {"median_ms": {key: value["median_ms"] for key, value in stages.items()}}),
                      flush=True)
    save()


if __name__ == "__main__":
    main()
