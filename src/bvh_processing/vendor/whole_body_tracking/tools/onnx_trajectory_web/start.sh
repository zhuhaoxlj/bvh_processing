#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
python_bin="${ONNX_TRAJECTORY_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  for candidate in "${VIRTUAL_ENV:-}/bin/python" "$HOME/Project/01-RL/IsaacLab/env_isaaclab/bin/python" "$HOME/IsaacLab/env_isaaclab/bin/python" "$project_root/.venv/bin/python"; do
    if [[ -x "$candidate" ]]; then python_bin="$candidate"; break; fi
  done
fi
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then echo "找不到包含 onnxruntime 的 Python，请设置 ONNX_TRAJECTORY_PYTHON。" >&2; exit 1; fi
cd "$project_root"
exec "$python_bin" "$script_dir/server.py" "$@"
