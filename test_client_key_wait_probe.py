from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent / "benchmarks"))
from client_key_wait_probe import run_probe


def test_completed_transcript_waits_for_second_held_recording():
    project = Path.home() / "Projects/whisper-local"
    if not (project / "src/whisper_local/app.py").exists():
        pytest.skip("This integration probe needs the separate dictation client")
    result = run_probe(project, held_seconds=60)
    assert result["real_model_calls"] == 0
    assert result["while_key_held"] == {"model_finished": True, "inserted_count": 0, "model_call_count": 1}
    assert result["first_result_wait_after_model_s"] == 60
    assert [row["text"] for row in result["insertions"]] == ["transcript 0", "transcript 1"]
