#!/bin/bash
# 条件 A: チェックを渡さず、終わったら報告させる
source "$(dirname "$0")/common.sh"
setup "a$SUFFIX"

claude -p "SPEC.md のとおりに動くように src/billing.ts を直してください。終わったら報告してください。" \
  "${CLAUDE_FLAGS[@]}" < /dev/null > log/maker.json

summary log/maker.json
jq -r .result log/maker.json > log/report.md
