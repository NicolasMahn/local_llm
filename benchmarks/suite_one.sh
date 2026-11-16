#!/usr/bin/env bash
# Run the serving suite on one stopped model: start it, measure, stop it again.
#
#   benchmarks/suite_one.sh NAME LABEL OUTPUT_DIR [suite options...]
#
# Without --case among the options it runs the extended suite.
#
# Launch settings come from the environment, exactly as for ./llm up, e.g.
#   LOCAL_LLM_FAST_KERNELS=0 benchmarks/suite_one.sh gemma-4-31b native logs/x
# A model that already runs belongs to someone else: it is measured in place
# with added_models.py instead, never restarted from here.
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")/..
name=$1 label=$2 out=$3
shift 3
mkdir -p "$out"

if [[ $(podman inspect -f '{{.State.Running}}' "llm-$name" 2>/dev/null) == true ]]; then
  echo "$name is already running; measure it in place with benchmarks/added_models.py" >&2
  exit 1
fi

stop() { "$HERE/llm" stop "$name"; }
trap stop EXIT

"$HERE/llm" up "$name" >"$out/$name-$label.start.log" 2>&1 || {
  podman logs "llm-$name" >"$out/$name-$label.startup.log" 2>&1 || true
  echo "$name did not start; see $out/$name-$label.start.log" >&2
  exit 1
}
podman logs "llm-$name" >"$out/$name-$label.startup.log" 2>&1

# A model that keeps the GPU busy without requests costs power all day and
# leaves the suite no quiet window; record it before any request is sent.
sleep 20
python3 - "$out/$name-$label.idle.json" <<'PY'
import json, statistics, sys, time
from pathlib import Path

busy, power = [], []
for _ in range(10):
    busy.append(max(float(p.read_text()) for p in Path("/sys/class/drm").glob("card*/device/gpu_busy_percent")))
    stats = json.loads((Path.home() / ".local/state/local-llm/stats.json").read_text())
    power.append(stats["power_w"])
    time.sleep(1)
Path(sys.argv[1]).write_text(json.dumps({"idle_gpu_busy_percent": busy, "median_idle_gpu_busy_percent": statistics.median(busy),
                                         "package_power_w": power, "median_package_power_w": statistics.median(power)}) + "\n")
PY

cases=(--case extended)
[[ " $* " == *" --case "* ]] && cases=()
status=0
python3 "$HERE/benchmarks/added_models.py" "$name" "$label" "${cases[@]}" --keep-going --timeout 900 \
  --output "$out/$name-$label.json" "$@" >"$out/$name-$label.log" 2>&1 || status=$?
# Which local kernels the server really used is only known after the requests.
podman logs "llm-$name" 2>&1 | grep -E "Using local|native fallback" >"$out/$name-$label.kernels.log" || true
exit $status
