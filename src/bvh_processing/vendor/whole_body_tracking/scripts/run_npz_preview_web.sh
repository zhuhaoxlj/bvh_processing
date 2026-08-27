#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"

if [[ -n "${MOTION_PREVIEW_VENV:-}" ]]; then
  motion_preview_venv="$MOTION_PREVIEW_VENV"
elif [[ -n "${VIRTUAL_ENV:-}" && -d "$VIRTUAL_ENV" ]]; then
  motion_preview_venv="$VIRTUAL_ENV"
else
  motion_preview_venv=""
  for candidate in \
    "$HOME/Project/01-RL/IsaacLab/env_isaaclab" \
    "$HOME/IsaacLab/env_isaaclab" \
    "$project_root/.venv"; do
    if [[ -d "$candidate" ]]; then
      motion_preview_venv="$candidate"
      break
    fi
  done
fi

if [[ -z "$motion_preview_venv" || ! -d "$motion_preview_venv" ]]; then
  echo "Isaac Lab environment not found." >&2
  echo "Set MOTION_PREVIEW_VENV to the correct environment path." >&2
  exit 1
fi

python_bin="${MOTION_PREVIEW_PYTHON:-$motion_preview_venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Isaac Lab Python is not executable: $python_bin" >&2
  echo "Set MOTION_PREVIEW_PYTHON to the correct Python executable." >&2
  exit 1
fi

export VIRTUAL_ENV="$motion_preview_venv"
export PYTHONPATH="$project_root/source/whole_body_tracking${PYTHONPATH:+:$PYTHONPATH}"

cd "$project_root"
exec "$python_bin" scripts/npz_preview_web.py "$@"
