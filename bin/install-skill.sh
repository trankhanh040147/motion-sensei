#!/usr/bin/env bash
# Link the skill into ~/.agents/skills (shared) and ~/.claude/skills, then run setup.
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
NAME=basketball-form-coach
mkdir -p "$HOME/.agents/skills" "$HOME/.claude/skills"
ln -sfn "$ROOT/skills/$NAME" "$HOME/.agents/skills/$NAME"
[ -e "$HOME/.claude/skills/$NAME" ] || ln -s "../../.agents/skills/$NAME" "$HOME/.claude/skills/$NAME"
"$ROOT/skills/$NAME/scripts/setup.sh"
echo "installed: $HOME/.agents/skills/$NAME -> $ROOT/skills/$NAME"
