#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 2
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "A Conda environment must be activated before running this wrapper." >&2
  exit 2
fi

# Some shared clusters inject an older system libstdc++ ahead of the active
# Conda environment, which breaks Pillow/MMCV imports. Prepend the env lib dir
# so the runtime loader resolves Conda's C++ runtime first.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "$@"
