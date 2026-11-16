#!/usr/bin/env bash
# Fetch all configured weights, or only the named models. Finished files are
# reused and interrupted transfers resume; pinned additions retain their revision.
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")
exec python3 "$HERE/download_models.py" "$@"
