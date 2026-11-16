"""Check the context boundary and one near-limit synthetic capacity request.

This is an allocation/serving check, not a long-context accuracy benchmark.
"""

import argparse
import fcntl
import json
from pathlib import Path
import sys
import subprocess
import time
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.added_models import Guard, HERE, serving_metadata, timestamp
from model_catalog import MODELS


def resources():
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return {key: int(memory[key].split()[0]) * 1024 for key in ('MemAvailable', 'SwapFree')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=MODELS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-idle-gpu-busy', type=float, default=20)
    args = parser.parse_args()
    context = MODELS[args.model]['context']
    config = json.loads(subprocess.check_output([str(HERE/'llm'), 'config'], text=True))
    backends = {m['name']: f'http://127.0.0.1:{m["port"]}/v1' for m in config['models']}
    key = (Path.home()/'.config/local-llm/api-key').read_text().strip()
    guard = Guard(args.model, backends, args.max_idle_gpu_busy)
    report = {'at': timestamp(), 'runtime': serving_metadata(args.model), 'context': context,
              'kind': 'synthetic capacity; no accuracy claim', 'resources_before': resources()}

    def send(tokens):
        body = {'model': args.model, 'prompt': [100]*tokens, 'max_tokens': 1,
                'temperature': 0, 'cache_salt': uuid.uuid4().hex}
        req = urllib.request.Request(backends[args.model]+'/completions',
            data=json.dumps(body).encode(), headers={'Content-Type':'application/json',
                                                    'Authorization':'Bearer '+key})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                return {'status': response.status, 'elapsed_s':time.monotonic()-started,
                        'response': json.load(response)}
        except urllib.error.HTTPError as exc:
            return {'status': exc.code, 'elapsed_s':time.monotonic()-started,
                    'response': json.loads(exc.read())}

    with Path('/run/user/1000/local-llm-benchmark.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def trial(tokens, count, expected_status):
            attempts = []
            for _ in range(64):
                value,audit = guard.trial(lambda: send(tokens),count)
                attempts.append(audit)
                if value is not None and value['status'] != expected_status:
                    return value,audit,attempts
                if not audit['reasons']:
                    return value,audit,attempts
                time.sleep(0.5)
            raise SystemExit('No clean context-check window: '+str(attempts[-1]))

        value,audit,attempts = trial(context+1,0,400)
        report['over_limit'] = {'result':value,'audit':audit,'attempts':attempts}
        if audit['reasons'] or value['status'] != 400 or 'context' not in str(value['response']).lower():
            raise SystemExit('Context boundary failed: '+str(report['over_limit']))
        value,audit,attempts = trial(context-256,1,200)
        report['near_limit'] = {'result':value,'audit':audit,'attempts':attempts}
        report['resources_after'] = resources()
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        if audit['reasons'] or value['status'] != 200:
            raise SystemExit('Near-limit request failed; see '+str(args.output))


if __name__ == '__main__':
    main()
