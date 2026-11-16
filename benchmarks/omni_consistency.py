"""Repeat Omni workloads, including whisper-local's actual dictation contract.

Only synthetic recordings are used. Nothing is recorded, pasted or sent to a
desktop application. Model settings and client safety prompts stay unchanged.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from unittest.mock import patch
import urllib.request
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import media_part, request
from consistency_stats import numeric_summary, response_summary
from omni_inputs import InputGuard


SHORT = "The deployment report is ready. Please check job 4711 and confirm there are 0 errors."
LONG = (
    "Today I am testing the dictation app with a longer recording. "
    "I want the words to appear exactly as I say them. "
    "The first task is to check the report before the meeting. "
    "The second task is to review the changes and keep a note of anything that looks unusual. "
    "Please do not send a message or run a command. "
    "Just write down this sentence, including the question: when will the next build finish? "
    "I am leaving a short pause now, and then I will continue speaking. "
    "Some recordings contain many words, while others contain long pauses. "
    "Those should be measured separately. "
    "At the end of this test, the job number is 4711 and the error count is 0. "
    "This is the final sentence of the recording."
)


def audio_seconds(wav):
    with wave.open(io.BytesIO(wav), "rb") as clip:
        return clip.getnframes() / clip.getframerate()


def pad_recording(wav, seconds):
    with wave.open(io.BytesIO(wav), "rb") as clip:
        params, samples = clip.getparams(), clip.readframes(clip.getnframes())
    frames = len(samples) // (params.sampwidth * params.nchannels)
    output = io.BytesIO()
    with wave.open(output, "wb") as clip:
        clip.setparams(params)
        clip.writeframes(samples + b"\0" * max(0, round(seconds * params.framerate) - frames)
                         * params.sampwidth * params.nchannels)
    return output.getvalue()


def prepare_fixtures(directory, screenshot):
    directory.mkdir(parents=True, exist_ok=True)
    recordings = {}
    for name, text, speed in (("short", SHORT, 175), ("dense", LONG, 155)):
        source, target = directory / f"{name}-source.wav", directory / f"{name}.wav"
        subprocess.run(["espeak-ng", "-s", str(speed), "-w", str(source), text], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-ar", "16000",
                        "-ac", "1", "-c:a", "pcm_s16le", str(target)], check=True)
        recordings[name] = target.read_bytes()
    duration = max(60, audio_seconds(recordings["dense"]))
    recordings["dense"] = pad_recording(recordings["dense"], duration)
    recordings["sparse"] = pad_recording(recordings["short"], duration)
    for name, wav in recordings.items():
        (directory / f"{name}.wav").write_bytes(wav)
    (directory / "screen.png").write_bytes(screenshot)
    return recordings


def media_identity(body, nonce):
    """Force cold caches without changing image pixels or decoded audio.

    The same audio in the screenshot safety pass shares an identity with its
    audio-only pass. That preserves the real client's within-job cache reuse.
    """
    for message in body["messages"]:
        if not isinstance(message["content"], list):
            continue
        for part in message["content"]:
            if part.get("type") in ("image_url", "audio_url", "input_audio"):
                content = json.dumps({k: v for k, v in part.items() if k != "uuid"}, sort_keys=True)
                part["uuid"] = nonce + "-" + hashlib.sha256(content.encode()).hexdigest()
    return body


class ClientProbe:
    """Observe the real non-streaming client in this benchmark process only."""
    def __init__(self, client, nonce):
        self.client, self.nonce = client, nonce
        self.original = client._post
        self.rows = []
        self.lock = threading.Lock()

    def __call__(self, original_request, timeout):
        body = media_identity(json.loads(original_request.data), self.nonce)
        outgoing = urllib.request.Request(original_request.full_url, data=json.dumps(body).encode(),
                                          headers=dict(original_request.header_items()), method="POST")
        began = time.perf_counter()
        row = {"ttft_s": None, "total_s": None, "answer": "", "usage": {}, "finish_reason": None,
               "arm": "screenshot" if any(isinstance(m["content"], list) and
                   any(p.get("type") == "image_url" for p in m["content"]) for m in body["messages"]) else "audio"}
        try:
            payload = self.original(outgoing, timeout)
            result = json.loads(payload)
            choice = result["choices"][0]
            row.update(answer=self.client._chat_text(payload), usage=result.get("usage", {}),
                       finish_reason=choice.get("finish_reason"))
            return payload
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            row["total_s"] = time.perf_counter() - began
            with self.lock:
                self.rows.append(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--client-project", type=Path, default=Path.home() / "Projects/whisper-local")
    parser.add_argument("--screenshot", type=Path, default=Path("logs/performance-2026-11-02/fixtures/screen.png"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--parallel-repeats", type=int, default=3)
    parser.add_argument("--only", nargs="+", help="Run only named cases")
    args = parser.parse_args()
    if min(args.repeats, args.parallel_repeats) < 1:
        parser.error("Repetition counts must be positive")
    sys.path.insert(0, str(args.client_project / "src"))
    from whisper_local import transcribe as client
    from whisper_local.app import _start_worker
    from whisper_local.config import load

    settings = replace(load(), url="http://127.0.0.1:8400/v1", model="qwen3-omni-30b")
    if not settings.api_key:
        settings = replace(settings, api_key=(Path.home() / ".config/local-llm/api-key").read_text().strip())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    picture = args.screenshot.read_bytes()
    recordings = prepare_fixtures(args.output.parent / "fixtures", picture)
    from PIL import Image, ImageOps
    source_image = Image.open(io.BytesIO(picture)).convert("RGB")
    large = source_image.resize((3840, 2160), Image.Resampling.LANCZOS)
    portrait = ImageOps.pad(source_image, (1080, 1920), Image.Resampling.LANCZOS)
    geometry_parts = {}
    for name, resized in (("image_4k", large), ("image_portrait", portrait)):
        path = args.output.parent / "fixtures" / f"{name}.png"
        resized.save(path)
        geometry_parts[name] = media_part(path, "image")
    guard = InputGuard(["http://127.0.0.1:8006/v1", "http://127.0.0.1:8002/v1"])
    report = {"at": datetime.now(timezone.utc).isoformat(), "model": settings.model,
              "client_source_sha256": hashlib.sha256(Path(client.__file__).read_bytes()).hexdigest(),
              "client_language": settings.language, "vocabulary_entries": len(settings.vocabulary),
              "image_geometries": {"original": source_image.size, "image_4k": large.size, "image_portrait": portrait.size},
              "audio_seconds": {k: audio_seconds(v) for k, v in recordings.items()},
              "synthetic_inputs_only": True, "trials": [], "summaries": {}}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def chat_body(content, maximum=48):
        return {"model": settings.model, "messages": [{"role": "user", "content": content}],
                "temperature": 0, "max_tokens": maximum, "stream": True,
                "stream_options": {"include_usage": True}}

    def dictation_body(wav):
        return {"model": settings.model, "messages": client._chat_messages(wav, settings, None),
                "temperature": 0, "max_tokens": 64 + int(16 * len(wav) / 32000),
                "stream": True, "stream_options": {"include_usage": True}}

    def audio_part(wav):
        return {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}}

    def stream_calls(bodies):
        began = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(bodies)) as pool:
            futures = [pool.submit(request, settings.url, settings.api_key, body, 120) for body in bodies]
            rows = [future.result() for future in futures]
        return {"rows": rows, "job_elapsed_s": time.perf_counter() - began}

    def client_call(wav, screenshot, nonce):
        probe = ClientProbe(client, nonce)
        began = time.perf_counter()
        with patch.object(client, "_post", probe):
            try:
                answer = client.transcribe_chat(wav, settings, screenshot)
                error = None
            except Exception as exc:
                answer, error = None, f"{type(exc).__name__}: {exc}"
        return {"rows": probe.rows, "job_elapsed_s": time.perf_counter() - began,
                "final_answer": answer, "job_error": error}

    def queued_client_calls(nonce):
        probe = ClientProbe(client, nonce)
        submitted = time.perf_counter()
        jobs, events = [], [threading.Event(), threading.Event()]
        submit = _start_worker()

        def job(index, wav):
            began = time.perf_counter()
            result = {"queue_wait_s": began - submitted}
            try:
                result["answer"] = client.transcribe_chat(wav, settings, None)
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            result.update(processing_s=time.perf_counter() - began, ready_s=time.perf_counter() - submitted)
            jobs.append({"job": index, **result})
            events[index].set()

        with patch.object(client, "_post", probe):
            submit(lambda: job(0, recordings["dense"]))
            submit(lambda: job(1, recordings["short"]))
            for event in events:
                if not event.wait(150):
                    raise RuntimeError("Client queue probe did not complete")
        return {"rows": probe.rows, "jobs": jobs, "job_elapsed_s": time.perf_counter() - submitted}

    image = media_part(args.screenshot, "image")
    qa = "Give the job number and error count using digits in one short sentence."
    cases = [
        ("text_repeated", 1, 0, True, None, lambda nonce: stream_calls([chat_body(
            "Explain how a compiler works in detailed full paragraphs.", 256)])),
        ("text", 1, 0, False, None, lambda nonce: stream_calls([chat_body(
            f"Case {nonce}. Explain how a compiler works in detailed full paragraphs.", 256)])),
        ("image_fresh", 1, 1, False, None, lambda nonce: stream_calls([media_identity(chat_body(
            [image.copy(), {"type": "text", "text": qa}]), nonce)])),
        ("image_cached", 1, 1, True, None, lambda nonce: stream_calls([media_identity(chat_body(
            [image.copy(), {"type": "text", "text": qa}]), nonce)])),
        ("image_4k", 1, 1, False, None, lambda nonce: stream_calls([media_identity(chat_body(
            [geometry_parts["image_4k"].copy(), {"type": "text", "text": qa}]), nonce)])),
        ("image_portrait", 1, 1, False, None, lambda nonce: stream_calls([media_identity(chat_body(
            [geometry_parts["image_portrait"].copy(), {"type": "text", "text": qa}]), nonce)])),
        ("audio_qa_long", 1, 1, False, None, lambda nonce: stream_calls([media_identity(chat_body(
            [{"type": "text", "text": qa}, audio_part(recordings["dense"])]), nonce)])),
        ("dictation_short", 1, 1, False, SHORT, lambda nonce: stream_calls([
            media_identity(dictation_body(recordings["short"]), nonce)])),
        ("dictation_long", 1, 1, False, LONG, lambda nonce: stream_calls([
            media_identity(dictation_body(recordings["dense"]), nonce)])),
        ("dictation_long_cached", 1, 1, True, LONG, lambda nonce: stream_calls([
            media_identity(dictation_body(recordings["dense"]), nonce)])),
        ("dictation_sparse", 1, 1, False, SHORT, lambda nonce: stream_calls([
            media_identity(dictation_body(recordings["sparse"]), nonce)])),
        ("combined_qa", 1, 2, False, None, lambda nonce: stream_calls([media_identity(chat_body(
            [image.copy(), {"type": "text", "text": qa}, audio_part(recordings["short"])]), nonce)])),
        ("client_no_image", 1, 1, False, LONG, lambda nonce: client_call(recordings["dense"], None, nonce)),
        ("client_screenshot_safety", 2, 2, False, LONG, lambda nonce: client_call(recordings["dense"], picture, nonce)),
        ("parallel_dictation", 2, 2, False, LONG, lambda nonce: stream_calls([
            media_identity(dictation_body(recordings["dense"]), nonce + str(i)) for i in range(2)])),
        ("parallel_mixed", 2, 1, False, None, lambda nonce: stream_calls([
            chat_body(f"Case {nonce}. Explain how a compiler works in detailed full paragraphs.", 256),
            media_identity(chat_body([image.copy(), {"type": "text", "text": qa}]), nonce)])),
        ("client_queue", 2, 2, False, None, queued_client_calls),
    ]
    names = {case[0] for case in cases}
    if args.only and set(args.only) - names:
        parser.error(f"Unknown cases: {set(args.only) - names}")
    for name, count, unique_media, cached, reference, call in cases:
        if args.only and name not in args.only:
            continue
        repeats = args.parallel_repeats if count > 1 else args.repeats
        accepted, attempts = [], 0
        cached_nonce = uuid.uuid4().hex
        while len(accepted) < repeats + 1:
            if attempts >= repeats + 20:
                raise RuntimeError(f"No clean window for {name}; partial report saved")
            nonce = cached_nonce if cached else uuid.uuid4().hex
            began = time.perf_counter()
            value, audit = guard.trial(lambda: call(nonce), count)
            attempts += 1
            trial = {"case": name, "warmup": not accepted, "at": datetime.now(timezone.utc).isoformat(),
                     "audit": audit, "result": value, "observed_trial_s": time.perf_counter() - began}
            report["trials"].append(trial)
            if not audit["reasons"]:
                before, after = audit["before"][0]["input_counters"], audit["after"][0]["input_counters"]
                delta = {k: after[k] - before.get(k, 0) for k in after}
                queries, hits = delta.get("vllm:mm_cache_queries_total", 0), delta.get("vllm:mm_cache_hits_total", 0)
                if unique_media and not cached and queries - hits < unique_media:
                    audit["reasons"].append("Fresh trial did not miss every unique media identity")
                if unique_media and cached and accepted and hits != queries:
                    audit["reasons"].append("Cached trial included a media miss")
                value["backend_counter_delta"] = delta
            save()
            if audit["reasons"]:
                print(json.dumps({"case": name, "discarded": audit["reasons"]}), flush=True)
                time.sleep(.5)
                continue
            accepted.append(value)
            print(json.dumps({"case": name, "warmup": len(accepted) == 1,
                              "job_s": value["job_elapsed_s"],
                              "ttft_s": [r.get("ttft_s") for r in value["rows"]],
                              "total_s": [r.get("total_s") for r in value["rows"]],
                              "output_tokens": [r.get("usage", {}).get("completion_tokens") for r in value["rows"]],
                              "job_error": value.get("job_error")}), flush=True)
        measured = accepted[1:]
        rows = [row for value in measured for row in value["rows"]]
        summary = response_summary(rows, reference=reference)
        summary.update(job_elapsed_s=numeric_summary([v["job_elapsed_s"] for v in measured]),
                       discarded_trials=attempts-len(accepted), expected_requests_per_trial=count,
                       intentional_output_cap=name in ("text", "text_repeated", "parallel_mixed"),
                       text_prompt_varies=name in ("text", "parallel_mixed"))
        if name in ("image_fresh", "image_cached", "image_4k", "image_portrait", "audio_qa_long", "combined_qa"):
            summary["factual_answers_correct"] = sum("4711" in (r.get("answer") or "") and
                                                     "0" in (r.get("answer") or "") for r in rows)
        if name.startswith("parallel_"):
            summary["request_slots"] = [response_summary([value["rows"][i] for value in measured], reference=reference)
                                        for i in range(count)]
        if name == "client_screenshot_safety":
            summary["request_arms"] = {arm: response_summary([r for r in rows if r["arm"] == arm], reference=reference)
                                       for arm in ("audio", "screenshot")}
        if name in ("client_no_image", "client_screenshot_safety"):
            summary["client_final_answers"] = response_summary([
                {"answer": value["final_answer"], "total_s": value["job_elapsed_s"],
                 "error": value["job_error"]} for value in measured], reference=reference)
        if name == "client_queue":
            summary["queue_jobs"] = [{"job": i, **{field: numeric_summary([next(j for j in v["jobs"] if j["job"] == i)[field] for v in measured])
                                     for field in ("queue_wait_s", "processing_s", "ready_s")}} for i in (0, 1)]
        report["summaries"][name] = summary
        save()
        print(json.dumps({"case": name, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
