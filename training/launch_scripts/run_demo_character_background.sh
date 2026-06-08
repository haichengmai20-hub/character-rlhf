#!/bin/bash
set -euo pipefail

source /home/ps/miniconda3/etc/profile.d/conda.sh
conda activate gemma4-character
cd /home/ps/hanjiaqi/LlamaFactory/chatjoy
export PYTHONNOUSERSITE=1
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
export no_proxy="localhost,127.0.0.1,0.0.0.0"

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

python - <<'PY'
import importlib.util
import gradio as gr

orig_launch = gr.Blocks.launch


def patched_launch(self, *args, **kwargs):
    kwargs["share"] = False
    kwargs.setdefault("server_name", "0.0.0.0")
    kwargs.setdefault("server_port", 7860)
    return orig_launch(self, *args, **kwargs)


gr.Blocks.launch = patched_launch

script_path = "/home/ps/hanjiaqi/LlamaFactory/chatjoy/demo_character.py"
spec = importlib.util.spec_from_file_location("demo_character_runtime", script_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.main()
PY
