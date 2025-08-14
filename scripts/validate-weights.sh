#!/usr/bin/env bash
set -euo pipefail
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="${HERE}/.."
python "${ROOT}/configs/symbolic/overrides/weights/validate.py" "$@"
