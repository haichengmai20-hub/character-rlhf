#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-gemma4-character}"
PROJECT_DIR="/home/ps/hanjiaqi/LlamaFactory/chatjoy"

source /home/ps/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

export PYTHONNOUSERSITE=1
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
export no_proxy="${NO_PROXY}"

PY_MM="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
SITE_PACKAGES="${CONDA_PREFIX}/lib/python${PY_MM}/site-packages"
EXTRA_LIBS=""
for d in "${SITE_PACKAGES}"/nvidia/*/lib; do
  if [ -d "${d}" ]; then
    EXTRA_LIBS="${EXTRA_LIBS:+${EXTRA_LIBS}:}${d}"
  fi
done
if [ -n "${EXTRA_LIBS}" ]; then
  export LD_LIBRARY_PATH="${EXTRA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

cd "${PROJECT_DIR}"
python "${PROJECT_DIR}/demo_character.py"
