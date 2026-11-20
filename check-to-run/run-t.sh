#!/bin/bash
# 追加の実験: 仕様だけを見せてテストを書かせ、バグ入りの実装と直した実装の両方に当てる
# 先に run-c.sh を実行して、直した実装（runs/c）を用意しておく。
source "$(dirname "$0")/common.sh"
FIXED="$ROOT/runs/c$SUFFIX/src/billing.ts"
[ -f "$FIXED" ] || { echo "先に run-c.sh を実行してください（$FIXED が必要です）" >&2; exit 1; }
setup "t$SUFFIX"
rm -rf src   # 作業フォルダから実装を消す（元の実装は base/ に残る）

claude -p "SPEC.md だけを読んで、test/billing.test.ts に node:test のテストを書いてください。実装はまだありません（../src/billing.ts から import する想定です）。仕様の各項目と、間違えやすい境界の値を確かめてください。テストは実行しないでください。" \
  "${CLAUDE_FLAGS[@]}" < /dev/null > log/maker.json
summary log/maker.json

# 書かれたテストを、それぞれの実装に当てる
mkdir -p src
for impl in "$ROOT/base/src/billing.ts" "$FIXED"; do
  cp "$impl" src/billing.ts
  for tz in Asia/Tokyo America/Los_Angeles; do
    result="$(TZ="$tz" node --test "test/*.test.ts" 2>&1 | grep -E '^ℹ (tests|pass|fail)' | tr '\n' ' ' || true)"
    echo "${impl#"$ROOT"/} TZ=$tz: $result"
  done
done
