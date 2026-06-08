#!/usr/bin/env python3
"""
merge_gemma4_sft_lora.py
将 LlamaFactory 训练的 Gemma4 SFT LoRA adapter 合并到 base Gemma4 模型，
输出 HF 格式 Gemma4ForConditionalGeneration 完整权重。

用法:
  conda run -n gemma4-rl python3 merge_gemma4_sft_lora.py

输出目录: saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072_merged/
"""

import json
import shutil
from pathlib import Path

import torch
import safetensors.torch as st

# ── 路径配置 ─────────────────────────────────────────────────────────────────
BASE_MODEL = Path("/home/ps/hanjiaqi/LlamaFactory/chatjoy/models/gemma4-26b-a4b-it-textonly")
ADAPTER    = Path("/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072")
OUTPUT     = Path("/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/sft2_ddp_lora_3gpu_567_cut3072_merged")

# LoRA 超参（与训练时一致）
LORA_ALPHA = 64
LORA_RANK  = 32
SCALING    = LORA_ALPHA / LORA_RANK  # 2.0


def load_adapter(adapter_path: Path) -> dict[str, torch.Tensor]:
    p = adapter_path / "adapter_model.safetensors"
    print(f"[merge] loading adapter from {p} ...")
    tensors = st.load_file(str(p))
    print(f"[merge] adapter keys: {len(tensors)}")
    return tensors


def build_lora_delta_map(adapter: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Δ = lora_B @ lora_A × scaling
    adapter key:  base_model.model.model.language_model.layers.0.self_attn.k_proj.lora_A.weight
    base key:     model.language_model.layers.0.self_attn.k_proj.weight
    """
    deltas: dict[str, torch.Tensor] = {}
    a_keys = [k for k in adapter if "lora_A" in k]
    print(f"[merge] found {len(a_keys)} lora_A keys → {len(a_keys)} deltas to compute")
    for a_key in a_keys:
        b_key = a_key.replace("lora_A", "lora_B")
        if b_key not in adapter:
            print(f"  [warn] missing lora_B for {a_key}, skipping")
            continue
        A = adapter[a_key]
        B = adapter[b_key]
        delta = (B.float() @ A.float()) * SCALING
        base_key = a_key.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
        deltas[base_key] = delta.to(A.dtype)
    return deltas


def process_shard(
    shard_path: Path,
    lora_deltas: dict[str, torch.Tensor],
    output_path: Path,
) -> list[str]:
    print(f"[merge] processing shard: {shard_path.name}  ({shard_path.stat().st_size / 1e9:.1f} GB)")
    tensors = st.load_file(str(shard_path))
    out_tensors: dict[str, torch.Tensor] = {}
    for name, w in tensors.items():
        if name in lora_deltas:
            out_tensors[name] = w + lora_deltas[name].to(w.device)
            print(f"  applied delta to {name}")
        else:
            out_tensors[name] = w

    st.save_file(out_tensors, str(output_path))
    return list(out_tensors.keys())


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    adapter = load_adapter(ADAPTER)
    lora_deltas = build_lora_delta_map(adapter)

    # 验证 delta key 能在 base model index 中找到
    index_file = BASE_MODEL / "model.safetensors.index.json"
    with open(index_file) as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]

    missing = [k for k in lora_deltas if k not in weight_map]
    if missing:
        print(f"[warn] {len(missing)} delta keys NOT found in base model weight_map:")
        for k in missing[:5]:
            print(f"  {k}")
        raise RuntimeError("Delta key mismatch — check adapter naming.")
    print(f"[merge] all {len(lora_deltas)} delta keys verified in base model ✓")

    shard_files = sorted(set(weight_map.values()))
    new_weight_map: dict[str, str] = {}

    for shard_fname in shard_files:
        src = BASE_MODEL / shard_fname
        dst = OUTPUT / shard_fname
        written_keys = process_shard(src, lora_deltas, dst)
        for k in written_keys:
            new_weight_map[k] = shard_fname

    new_index = {"metadata": index.get("metadata", {}), "weight_map": new_weight_map}
    with open(OUTPUT / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)
    print(f"[merge] wrote model.safetensors.index.json, {len(new_weight_map)} weights")

    # 复制所有配置和 tokenizer 文件
    copy_files = [
        "config.json", "generation_config.json", "tokenizer_config.json",
        "tokenizer.json", "chat_template.jinja", "processor_config.json",
    ]
    for fname in copy_files:
        src = BASE_MODEL / fname
        if src.exists():
            shutil.copy(src, OUTPUT / fname)
            print(f"[merge] copied {fname}")

    # 从 adapter 目录补充 tokenizer 文件（LlamaFactory 可能更新过）
    adapter_copy = ["tokenizer_config.json", "tokenizer.json", "chat_template.jinja"]
    for fname in adapter_copy:
        src = ADAPTER / fname
        if src.exists():
            shutil.copy(src, OUTPUT / fname)
            print(f"[merge] overwrite {fname} from adapter dir")

    print(f"\n[merge] Done → {OUTPUT}")


if __name__ == "__main__":
    main()
