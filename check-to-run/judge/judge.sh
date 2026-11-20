#!/bin/bash
# 採点用のテストを、2 つのタイムゾーンで実行する。どちらかで失敗すると exit 1。
# 使い方: judge/judge.sh base     （バグ入りの実装）
#         judge/judge.sh runs/a   （条件 A の結果）
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

[[ "${1:-}" =~ ^(base|runs/[a-z0-9-]+)$ ]] || { echo "使い方: judge/judge.sh base | runs/<name>" >&2; exit 1; }
export TARGET="$ROOT/$1/src/billing.ts"
[ -f "$TARGET" ] || { echo "$TARGET が見つかりません" >&2; exit 1; }

status=0
for tz in Asia/Tokyo America/Los_Angeles; do
  echo "== TZ=$tz"
  out="$(TZ="$tz" node --test "$ROOT/judge/judge.test.ts" 2>&1)" || status=1
  grep -E '^(✔|✖|ℹ (pass|fail))' <<< "$out" || true
done
exit "$status"
