import io
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import urllib.request
import wave

sys.path.insert(0, str(Path(__file__).parent / "benchmarks"))
from omni_consistency import ClientProbe, audio_seconds, media_identity, pad_recording


def test_fresh_media_identity_changes_cache_keys_without_changing_content():
    audio = {"type": "input_audio", "input_audio": {"data": "same-audio", "format": "wav"}}
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "same prompt"}, audio]}]}
    media_identity(body, "trial-a")
    first = audio["uuid"]
    assert audio["input_audio"] == {"data": "same-audio", "format": "wav"}
    assert body["messages"][0]["content"][0] == {"type": "text", "text": "same prompt"}
    media_identity(body, "trial-b")
    assert audio["uuid"] != first
    media_identity(body, "trial-a")
    assert audio["uuid"] == first


def test_safety_passes_share_audio_identity_within_one_job():
    def body(with_image):
        parts = [{"type": "input_audio", "input_audio": {"data": "wav", "format": "wav"}}]
        if with_image:
            parts.insert(0, {"type": "image_url", "image_url": {"url": "data:image/png;base64,png"}})
        return {"messages": [{"role": "user", "content": parts}]}
    audio_only = media_identity(body(False), "same-job")
    screenshot = media_identity(body(True), "same-job")
    assert audio_only["messages"][0]["content"][-1]["uuid"] == screenshot["messages"][0]["content"][-1]["uuid"]
    assert len({p["uuid"] for p in screenshot["messages"][0]["content"]}) == 2


def test_recording_padding_keeps_original_pcm_and_never_truncates():
    output = io.BytesIO()
    with wave.open(output, "wb") as clip:
        clip.setparams((1, 2, 10, 0, "NONE", "not compressed"))
        samples = struct.pack("<hhh", 1, -2, 3)
        clip.writeframes(samples)
    original = output.getvalue()
    padded = pad_recording(original, 1)
    assert audio_seconds(padded) == 1
    with wave.open(io.BytesIO(padded)) as clip:
        assert clip.readframes(10) == samples + b"\0" * 14
    assert pad_recording(original, .1) == original


def test_client_probe_preserves_request_contract_and_never_records_keys():
    seen = []
    reply = json.dumps({"choices": [{"message": {"content": "synthetic words"}, "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 2}}).encode()
    def post(request, timeout):
        seen.append((request, timeout))
        return reply
    client = SimpleNamespace(_post=post, _chat_text=lambda payload: json.loads(payload)["choices"][0]["message"]["content"])
    probe = ClientProbe(client, "test-job")
    body = {"temperature": 0, "max_tokens": 1024, "messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": "wav", "format": "wav"}}]}]}
    request = urllib.request.Request("http://localhost:8400/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer private-test-key", "Content-Type": "application/json"})
    assert probe(request, 90) == reply
    sent, timeout = seen[0]
    assert timeout == 90 and sent.get_header("Authorization") == "Bearer private-test-key"
    sent_body = json.loads(sent.data)
    assert sent_body["temperature"] == 0 and sent_body["max_tokens"] == 1024
    assert "stream" not in sent_body
    assert probe.rows[0]["arm"] == "audio" and probe.rows[0]["ttft_s"] is None
    assert probe.rows[0]["usage"] == {"completion_tokens": 2}
    assert "private-test-key" not in json.dumps(probe.rows)
