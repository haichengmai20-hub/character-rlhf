#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-gemma4-character}"
PROJECT_DIR="/home/ps/hanjiaqi/LlamaFactory/chatjoy"
REQ_FILE="${PROJECT_DIR}/requirements-gemma4-demo.txt"

source /home/ps/miniconda3/etc/profile.d/conda.sh

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.12
fi

conda activate "${ENV_NAME}"
export PYTHONNOUSERSITE=1

python -m pip install -U pip setuptools wheel
python -m pip install -r "${REQ_FILE}"

if ! python - <<'PY'
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
raise SystemExit(0 if "gemma4" in CONFIG_MAPPING else 1)
PY
then
  echo "Gemma 4 support not found in installed transformers; upgrading from official source..."
  python -m pip install -U "git+https://github.com/huggingface/transformers.git"
fi

PY_MM="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
SITE_PACKAGES="${CONDA_PREFIX}/lib/python${PY_MM}/site-packages"
ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/deactivate.d"

mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

cat > "${ACTIVATE_DIR}/gemma4_demo_env.sh" <<EOF
#!/usr/bin/env bash
export GEMMA4_DEMO_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH-}"
export GEMMA4_DEMO_OLD_PYTHONNOUSERSITE="\${PYTHONNOUSERSITE-}"
export PYTHONNOUSERSITE=1
_gemma4_libs=""
for d in "${SITE_PACKAGES}"/nvidia/*/lib; do
  if [ -d "\${d}" ]; then
    _gemma4_libs="\${_gemma4_libs:+\${_gemma4_libs}:}\${d}"
  fi
done
if [ -n "\${_gemma4_libs}" ]; then
  export LD_LIBRARY_PATH="\${_gemma4_libs}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
fi
unset _gemma4_libs
EOF

cat > "${DEACTIVATE_DIR}/gemma4_demo_env.sh" <<'EOF'
#!/usr/bin/env bash
if [ "${GEMMA4_DEMO_OLD_LD_LIBRARY_PATH+x}" = "x" ]; then
  export LD_LIBRARY_PATH="${GEMMA4_DEMO_OLD_LD_LIBRARY_PATH}"
else
  unset LD_LIBRARY_PATH
fi
if [ "${GEMMA4_DEMO_OLD_PYTHONNOUSERSITE+x}" = "x" ]; then
  export PYTHONNOUSERSITE="${GEMMA4_DEMO_OLD_PYTHONNOUSERSITE}"
else
  unset PYTHONNOUSERSITE
fi
unset GEMMA4_DEMO_OLD_LD_LIBRARY_PATH
unset GEMMA4_DEMO_OLD_PYTHONNOUSERSITE
EOF

source "${ACTIVATE_DIR}/gemma4_demo_env.sh"

python - <<'PY'
import importlib.util as u
mods = ["torch", "vllm", "gradio", "transformers", "deep_translator"]
print({m: bool(u.find_spec(m)) for m in mods})
PY

echo
echo "Environment ready: ${ENV_NAME}"
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
echo "Run demo with:"
echo "  bash ${PROJECT_DIR}/run_demo_character_gemma4.sh ${ENV_NAME}"
