#!/usr/bin/env bash
set -euo pipefail
bin/selftest.sh
bin/run-train.sh --max-epochs=1
bin/run-predict.sh
