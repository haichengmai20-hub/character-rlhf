#!/usr/bin/env python3
"""
Dequantize Gemma4-26B-A4B AWQ (compressed-tensors) → bf16 safetensors.

The AWQ model has:
  - attention q/k/v/o projections: INT4 (compressed-tensors pack-quantized)
  - MoE expert weights: already bf16 (not quantized per recipe.yaml)
  - vision encoder: bf16 (not quantized)

Loading via transformers + compressed-tensors library automatically dequantizes
INT4 attention weights to bf16 during load. We then save the clean bf16 model.

Original model is never modified.
"""

import os
import sys
import time

import torch
from transformers import AutoProcessor, AutoModelForCausalLM

AWQ_PATH = (
    "/home/ps/.cache/huggingface/hub/"
    "models--cyankiwi--gemma-4-26B-A4B-it-AWQ-4bit/snapshots/"
    "c5b961cae592047a529e32a8fd740998fbac16be"
)
OUT_PATH = (
    "/home/ps/.cache/huggingface/hub/"
    "models--gemma-4-26B-A4B-it-bf16/snapshots/main"
)

def main():
    print(f"[dequantize] Source : {AWQ_PATH}")
    print(f"[dequantize] Output : {OUT_PATH}")
    os.makedirs(OUT_PATH, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1: load model – transformers + compressed-tensors dequantizes  #
    #         INT4 attention weights to bf16 automatically on from_pretrained #
    # ------------------------------------------------------------------ #
    print("[dequantize] Loading model to CPU (this takes ~5-10 min) ...")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        AWQ_PATH,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )

    print(f"[dequantize] Load done in {(time.time()-t0)/60:.1f} min")

    # Verify no quantization config remains in the model
    if hasattr(model, "quantization_config") and model.quantization_config is not None:
        print(f"[dequantize] WARNING: quantization_config still set: {model.quantization_config}")
        print("[dequantize] Attempting to remove ...")
        model.quantization_config = None
        if hasattr(model, "config"):
            model.config.quantization_config = None

    # ------------------------------------------------------------------ #
    # Step 2: save as plain bf16 safetensors                              #
    # ------------------------------------------------------------------ #
    print(f"[dequantize] Saving bf16 model to {OUT_PATH} ...")
    t1 = time.time()

    model.save_pretrained(
        OUT_PATH,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    print(f"[dequantize] Save done in {(time.time()-t1)/60:.1f} min")

    # ------------------------------------------------------------------ #
    # Step 3: copy tokenizer / processor config files                     #
    # ------------------------------------------------------------------ #
    print("[dequantize] Copying tokenizer and config files ...")
    try:
        processor = AutoProcessor.from_pretrained(AWQ_PATH)
        processor.save_pretrained(OUT_PATH)
    except Exception as e:
        print(f"[dequantize] AutoProcessor failed ({e}), copying files manually ...")
        import shutil
        for fname in [
            "tokenizer.json", "tokenizer_config.json",
            "chat_template.jinja", "generation_config.json",
            "processor_config.json",
        ]:
            src = os.path.join(AWQ_PATH, fname)
            if os.path.exists(src):
                shutil.copy2(src, OUT_PATH)
                print(f"  copied {fname}")

    # ------------------------------------------------------------------ #
    # Step 4: patch config.json – remove quantization_config             #
    # ------------------------------------------------------------------ #
    import json
    config_path = os.path.join(OUT_PATH, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        if "quantization_config" in cfg:
            del cfg["quantization_config"]
            print("[dequantize] Removed quantization_config from config.json")
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    # Step 5: report                                                      #
    # ------------------------------------------------------------------ #
    total_bytes = sum(
        os.path.getsize(os.path.join(OUT_PATH, f))
        for f in os.listdir(OUT_PATH)
        if os.path.isfile(os.path.join(OUT_PATH, f))
    )
    print(f"\n[dequantize] Done. Total size: {total_bytes/1e9:.1f} GB")
    print(f"[dequantize] Total wall time: {(time.time()-t0)/60:.1f} min")
    print(f"[dequantize] Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
