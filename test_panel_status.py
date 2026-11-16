"""Exercise the panel's real polling logic with stand-ins for desktop services."""

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Panel discovery check needs Node.js")
def test_panel_discovers_added_and_future_models_from_cli_without_an_allowlist():
    source = (Path(__file__).parent / "panel/extension.js").read_text()
    body = re.search(r"async _load\(\) \{(.*?)\n    \}\n\n    _addCopyRow", source, re.S).group(1)
    names = ["existing", "granite-4.2-30b", "nemotron-3.5-lightning-30b-a3b",
             "olmo-3.1-32b-think", "future-model"]
    program = """
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const load = new AsyncFunction('run', 'GLib', 'POLL_SECONDS', 'logError', BODY);
const names = NAMES;
const found = [];
const panel = {
  _llm: 'llm', _addModelRow: name => found.push(name), _poll: async () => {},
  _openaiRow: {detail: {}}, _anthropicRow: {detail: {}},
  _keyRow: {detail: {}}, _chatRow: {detail: {}},
};
const run = async argv => argv[1] === 'list'
  ? names.map(name => `${name} off`).join('\\n')
  : argv[1] === 'url' ? 'http://example.test' : 'synthetic-key';
await load.call(panel, run, {timeout_add_seconds: () => 1}, 3, error => {throw error;});
console.log(JSON.stringify(found));
""".replace("BODY", json.dumps(body)).replace("NAMES", json.dumps(names))
    result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", program],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == names


@pytest.mark.skipif(shutil.which("node") is None, reason="Panel polling check needs Node.js")
def test_off_model_storage_reason_is_visible_and_never_overrides_a_live_model():
    source = (Path(__file__).parent / "panel/extension.js").read_text()
    method = re.search(r"async _pollOnce\(\) \{(.*?)\n    \}\n\n    _showProblems", source, re.S).group(1)
    states = re.search(r"const State = (\{.*?\n\});", source, re.S).group(1)
    program = """
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const poll = new AsyncFunction('readStats', 'State', BODY);
const State = STATES;
const model = {toggle: {state: false}, error: null, refused: null};
const shown = [];
const panel = {
  _models: new Map([['example', model]]), _starting: new Set(), _stopping: new Set(),
  _containers: async () => new Map(), _showMeasurements: () => {},
  _show: (entry, state, detail) => shown.push({state, detail}),
};
for (const stats of [
  {models: {example: {up: false}}, model_states: {example: {state: 'off', reason: 'Required archive is not mounted'}}},
  {models: {example: {up: true}}, model_states: {example: {state: 'off', reason: 'stale blocker'}}},
  {models: {example: {up: false}}, model_states: {example: {state: 'off'}}},
]) {
  await poll.call(panel, () => stats, State);
  shown.at(-1).on = model.toggle.state;
}
console.log(JSON.stringify(shown));
""".replace("BODY", json.dumps(method)).replace("STATES", states)
    result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", program],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == [
        {"state": "failed", "detail": "Required archive is not mounted", "on": False},
        {"state": "live", "detail": "ready", "on": True},
        {"state": "off", "detail": "off", "on": False},
    ]
