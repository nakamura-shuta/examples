# shellcheck shell=bash
# run-*.sh から読み込む共通の設定。
set -eu

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for cmd in claude jq node; do
  command -v "$cmd" > /dev/null || { echo "$cmd が見つかりません" >&2; exit 1; }
done

# 出力先の名前に使うので、英小文字・数字・ハイフンだけを受け付ける
SUFFIX="${SUFFIX:-}"
[[ "$SUFFIX" =~ ^[a-z0-9-]*$ ]] || { echo "SUFFIX は英小文字・数字・ハイフンだけにしてください" >&2; exit 1; }

# 個人の設定・プラグイン・MCP サーバーを読み込まずに実行する（条件をそろえるため）
CLAUDE_FLAGS=(--setting-sources "project,local" --strict-mcp-config --output-format json --permission-mode auto)
# MODEL=haiku SUFFIX=-haiku ./run-a.sh のように、モデルと出力先を変えられる
if [ -n "${MODEL:-}" ]; then CLAUDE_FLAGS+=(--model "$MODEL"); fi

# setup <name>: runs/<name> を作り直し、仕様とバグ入りの実装を置く
# 同じ名前で実行し直すと、前回の runs/<name> は消える
setup() {
  local dir="$ROOT/runs/$1"
  # runs がシンボリックリンクだと、リンク先のフォルダを消してしまうので止める
  [ ! -L "$ROOT/runs" ] || { echo "$ROOT/runs がシンボリックリンクです" >&2; exit 1; }
  rm -rf "$dir"
  mkdir -p "$dir/log" "$dir/.claude"
  cp -R "$ROOT/base/." "$dir/"
  # 個人用と、このリポジトリの外側にある CLAUDE.md を読み込まない
  jq -n --arg user "$HOME/.claude/CLAUDE.md" --arg parent "$(dirname "$ROOT")/CLAUDE.md" \
    '{claudeMdExcludes: [$user, $parent]}' > "$dir/.claude/settings.json"
  cd "$dir"
}

# summary <json>: ターン数・費用・時間を 1 行で表示する
summary() {
  jq -r '"turns=\(.num_turns) cost=$\(.total_cost_usd) sec=\(.duration_ms/1000|floor)"' "$1"
}
