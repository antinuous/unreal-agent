#!/bin/bash
set -euo pipefail
cp /app/model-requests.jsonl /logs/verifier/model-requests.jsonl
cp /app/tool-env /logs/verifier/tool-env
test "$(cat /app/answer)" = done
printf '1\n' > /logs/verifier/reward.txt
