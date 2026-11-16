"""Compare fresh and reused Omni inputs through the serving API."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request
import wave
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import media_part, parse_backend_metrics, request
from serving_compare import Guard


TIMERS = ("time_to_first_token_seconds", "request_prefill_time_seconds",
          "request_queue_time_seconds", "request_decode_time_seconds")
COUNTERS = tuple(f"vllm:{name}_{suffix}" for name in TIMERS for suffix in ("sum", "count")) + (
    "vllm:mm_cache_queries_total", "vllm:mm_cache_hits_total")


def input_counters(metrics):
    result = {}
    for line in metrics.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split()[0]
        if name in COUNTERS:
            result[name] = result.get(name, 0) + float(line.rsplit(None, 1)[-1])
    return result


def engine_measurements(before, after):
    delta = {name: after[name] - before.get(name, 0) for name in after}
    result = {}
    for name in TIMERS:
        total, count = f"vllm:{name}_sum", f"vllm:{name}_count"
        if delta.get(count) == 1:
            result[name] = delta[total]
    for name in ("mm_cache_queries_total", "mm_cache_hits_total"):
        if f"vllm:{name}" in delta:
            result[name] = delta[f"vllm:{name}"]
    return result


def cache_reasons(engine, *, media_count, reuse, warmed):
    if not media_count:
        return []
    hits = engine.get("mm_cache_hits_total")
    if hits is None:
        return ["Multimodal cache counters are missing"]
    if not reuse and hits:
        return ["Fresh input unexpectedly hit the multimodal cache"]
    if reuse and warmed and hits != media_count:
        return ["Reused input did not hit every expected multimodal cache entry"]
    return []


class InputGuard(Guard):
    def snapshot(self):
        readings = []
        for url in self.urls:
            with urllib.request.urlopen(url.removesuffix("/v1") + "/metrics", timeout=5) as response:
                metrics = response.read().decode()
            readings.append({**parse_backend_metrics(metrics), "input_counters": input_counters(metrics)})
        return readings


def convert_audio(source, target, *options):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source),
                    *options, str(target)], check=True)


def fixtures(source, target):
    """Prepare recordings before measuring; no audio is trimmed or denoised."""
    target.mkdir(parents=True, exist_ok=True)
    texts = {
        "short": "The job number is four seven one one. The error count is zero.",
        "long": ("Today we are checking a local assistant. Listen carefully to this status report. "
                 "The recording contains a job number and an error count. "
                 "Please remember both numbers and report them using digits. "
                 "The job number is four seven one one. The error count is zero."),
    }
    for label, text in texts.items():
        spoken, native = target / f"{label}-speech.wav", target / f"{label}.wav"
        subprocess.run(["espeak-ng", "-s", "175", "-w", str(spoken), text], check=True)
        convert_audio(spoken, native, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le")
    convert_audio(source / "question.wav", target / "question.wav",
                  "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le")


def varied_audio(source, target, variant, opus):
    with wave.open(str(source), "rb") as clip:
        params, samples = clip.getparams(), clip.readframes(clip.getnframes())
    # One extra 20 ms frame per variant changes decoded samples even with Opus.
    # The same variants are used in each comparison mode.
    silence = 320 * (variant + 1)
    with wave.open(str(target), "wb") as clip:
        clip.setparams(params)
        clip.writeframes(samples + b"\0" * silence * params.sampwidth * params.nchannels)
    duration = (len(samples) // (params.sampwidth * params.nchannels) + silence) / params.framerate
    if opus:
        encoded = target.with_suffix(".webm")
        convert_audio(target, encoded, "-ar", "48000", "-ac", "1", "-c:a", "libopus", "-b:a", "64k")
        return encoded, duration
    return target, duration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="Report label; does not change server settings")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, default=Path("logs/performance-2026-11-02/fixtures"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--image-variant-offset", type=int, default=117)
    parser.add_argument("--audio-variant-offset", type=int, default=0)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Use at least one measured repetition")
    from PIL import Image
    args.output.parent.mkdir(parents=True, exist_ok=True)
    directory = args.output.parent / f"{args.output.stem}-inputs"
    fixtures(args.fixture_dir, directory)
    original = Image.open(args.fixture_dir / "screen.png").convert("RGB")
    key = (Path.home() / ".config/local-llm/api-key").read_text().strip()
    url = "http://127.0.0.1:8006/v1"
    guard = InputGuard([url, "http://127.0.0.1:8002/v1"])
    cases = [
        ("chat_fresh", None, None, False, False),
        ("image_fresh", True, None, False, False),
        ("image_followup", True, None, False, True),
        ("voice_wav_fresh", None, "short", False, False),
        ("voice_opus_fresh", None, "short", True, False),
        ("voice_opus_long_fresh", None, "long", True, False),
        ("image_voice_fresh", True, "question", True, False),
        ("image_voice_followup", True, "question", True, True),
    ]
    report = {"at": datetime.now(timezone.utc).isoformat(), "label": args.label,
              "model": "qwen3-omni-30b", "fixture_directory": str(directory),
              "variants": {"image_offset": args.image_variant_offset, "audio_offset": args.audio_variant_offset},
              "trials": [], "summaries": {}}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for case_index, (name, image, audio, opus, reuse) in enumerate(cases):
        rows = []
        reused_body = None
        attempts = 0
        deadline = time.monotonic() + 240
        while len(rows) < args.repeats + 1:
            if attempts >= 30 or time.monotonic() > deadline:
                raise RuntimeError(f"No clean measurement window for {name}; partial report saved")
            variant = 0 if reuse else attempts
            attempts += 1
            parts, metadata = [], {}
            if image:
                picture = original.copy()
                identity = (args.image_variant_offset << 8) + case_index * 32 + variant
                picture.putpixel((picture.width - 1, picture.height - 1),
                                 (identity & 255, identity >> 8 & 255, identity >> 16 & 255))
                path = directory / f"{name}-{variant}.png"
                picture.save(path)
                parts.append(media_part(path, "image"))
                metadata["image_pixels"] = original.size
            prompt = "What is the job number and error count? Answer in one short sentence using digits."
            if audio:
                prompt = ("Answer the spoken question about this screenshot in one short sentence using digits."
                          if image else "Give the job number and error count spoken in the recording, using digits in one short sentence.")
            if not image and not audio:
                prompt = f"Independent case {args.image_variant_offset}.{variant}. What is 17 times 23? Return only the digits."
            parts.append({"type": "text", "text": prompt})
            if audio:
                path, duration = varied_audio(directory / f"{audio}.wav",
                                             directory / f"{name}-{variant}.wav",
                                             args.audio_variant_offset + variant, opus)
                parts.append(media_part(path, "audio"))
                metadata.update(audio_duration_s=duration, audio_format="opus48k" if opus else "pcm16k")
            body = {"model": "qwen3-omni-30b", "messages": [{"role": "user", "content": parts}],
                    "temperature": 0, "max_tokens": 48, "stream": True,
                    "stream_options": {"include_usage": True}}
            if reuse:
                # WebM can contain a fresh container ID after each encoding.
                # Reuse the actual payload bytes, not just the decoded samples.
                if reused_body is None:
                    reused_body = body
                body = reused_body
            result, audit = guard.trial(lambda: request(url, key, body, 120), 1)
            trial = {"case": name, "attempt": attempts - 1, "warmup": not rows,
                     "metadata": metadata, "audit": audit, "result": result}
            report["trials"].append(trial)
            save()
            if not audit["reasons"]:
                engine = engine_measurements(audit["before"][0]["input_counters"], audit["after"][0]["input_counters"])
                result["engine"] = engine
                audit["reasons"].extend(cache_reasons(engine, media_count=int(bool(image)) + int(bool(audio)),
                                                     reuse=reuse, warmed=bool(rows)))
                save()
            if audit["reasons"]:
                print(json.dumps({"case": name, "discarded": audit["reasons"]}), flush=True)
                time.sleep(0.5)
                continue
            answer = result["answer"]
            correct = answer.strip() == "391" if not image and not audio else "4711" in answer and "0" in answer
            if not correct:
                raise RuntimeError(f"Answer check failed for {name}; inspect saved answer")
            if "time_to_first_token_seconds" in engine:
                # Timer boundaries need not match CPU/GPU processing stages.
                result["client_minus_reported_ttft_s"] = result["ttft_s"] - engine["time_to_first_token_seconds"]
            rows.append(result)
            save()
            print(json.dumps({"case": name, "warmup": len(rows) == 1,
                              "ttft_s": result["ttft_s"], "total_s": result["total_s"],
                              "engine": engine}), flush=True)
        measured = rows[1:]
        summary = {"requests": len(measured), "discarded_attempts": attempts - len(rows),
                   "median_ttft_s": statistics.median(r["ttft_s"] for r in measured),
                   "median_total_s": statistics.median(r["total_s"] for r in measured),
                   "median_engine": {key: statistics.median(r["engine"][key] for r in measured)
                                     for key in set.intersection(*(set(r["engine"]) for r in measured))}}
        if all("client_minus_reported_ttft_s" in r for r in measured):
            summary["median_client_minus_reported_ttft_s"] = statistics.median(r["client_minus_reported_ttft_s"] for r in measured)
        report["summaries"][name] = summary
        report["fixture_hashes"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in directory.iterdir() if f.suffix in (".png", ".wav")}
        save()
        print(json.dumps({"case": name, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
