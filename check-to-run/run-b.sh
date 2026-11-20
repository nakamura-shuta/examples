#!/bin/bash
# 条件 B: 読むだけのレビュアーが承認するまで、レビューと修正を繰り返す（最大 5 往復）
# レビュアーも修正役もコードを実行できないよう、Bash を禁止する。
source "$(dirname "$0")/common.sh"
setup "b$SUFFIX"

REVIEW_PROMPT="あなたはコードレビュアーです。SPEC.md と src/billing.ts を読んでレビューしてください。コードは実行できません。問題点を番号付きで列挙してください。最後の行には、修正が必要なら CHANGES_REQUESTED、不要なら APPROVED とだけ書いてください。"

for i in 1 2 3 4 5; do
  claude -p "$REVIEW_PROMPT" "${CLAUDE_FLAGS[@]}" \
    --allowedTools "Read,Glob,Grep" --disallowedTools "Bash,Edit,Write,Agent,Skill,Workflow,NotebookEdit" \
    < /dev/null > "log/review-$i.json"
  jq -r .result "log/review-$i.json" > "log/review-$i.md"
  verdict="$(tail -n 1 "log/review-$i.md")"
  echo "round $i review: $(summary "log/review-$i.json") verdict=$verdict"
  if [ "$verdict" = "APPROVED" ]; then break; fi

  claude -p "次のレビュー指摘に対応して src/billing.ts を直してください。コードは実行できません。終わったら報告してください。

$(cat "log/review-$i.md")" "${CLAUDE_FLAGS[@]}" \
    --allowedTools "Read,Edit,Glob,Grep" --disallowedTools "Bash,Write,Agent,Skill,Workflow,NotebookEdit" \
    < /dev/null > "log/fix-$i.json"
  jq -r .result "log/fix-$i.json" > "log/fix-$i.md"
  cp src/billing.ts "log/billing-after-$i.ts"
  echo "round $i fix: $(summary "log/fix-$i.json") lines=$(wc -l < src/billing.ts)"
done
