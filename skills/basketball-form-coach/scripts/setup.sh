#!/usr/bin/env bash
# Idempotent setup: Python env (uv, 3.12) + YOLO weights. Safe to re-run.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

command -v ffmpeg >/dev/null || { echo "ffmpeg missing -> installing"; brew install ffmpeg; }
command -v uv >/dev/null || { echo "uv missing -> installing"; brew install uv; }

uv sync --project "$ROOT" --python 3.12 -q
mkdir -p "$ROOT/models"
cd "$ROOT/models"
POSE="${MS_POSE_MODEL:-yolo11s-pose.pt}"
BALL="${MS_BALL_MODEL:-yolo11s.pt}"
for w in "$POSE" "$BALL"; do
  if [ ! -f "$w" ]; then
    uv run --project "$ROOT" python -c "from ultralytics import YOLO; YOLO('$w')" >/dev/null
  fi
done
echo "ok: env=$ROOT/.venv models=$(ls "$ROOT/models" | tr '\n' ' ')"
