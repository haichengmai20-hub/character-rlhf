#!/usr/bin/env python3
"""
merge_rm_lora_v5_for_vllm.py
将 OpenRLHF 训练的 RM v5 LoRA adapter 合并到 base SFT 模型，
同时提取 score head，输出 vLLM Qwen3MoeForSequenceClassification 可直接加载的 HF 格式。

用法:
  conda run -n gemma4-rl python3 merge_rm_lora_v5_for_vllm.py

输出目录: saves/qwen3-30b-moe/chatjoy_rm_v5_merged_vllm/
  - model-000XX-of-000XX.safetensors   (合并后权重，含 score.weight)
  - model.safetensors.index.json       (更新 weight_map)
  - config.json                        (architectures → Qwen3MoeForSequenceClassification)
  - tokenizer_config.json 等           (从 base 复制)
"""

import json
import os
import shutil
from pathlib import Path

import torch
import safetensors.torch as st

# ── 路径配置 ─────────────────────────────────────────────────────────────────
BASE = Path("/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe")
BASE_MODEL = BASE / "clean32_for_sft2_fsdp_qlora_merged"
ADAPTER    = BASE / "chatjoy_rm_v5_gpu0234_20260421"
OUTPUT     = BASE / "chatjoy_rm_v5_merged_vllm"

# LoRA 超参（与训练时一致）
LORA_ALPHA = 128
LORA_RANK  = 64
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
    adapter key:  base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight
    base key:     model.layers.0.self_attn.k_proj.weight
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
    score_weight: torch.Tensor | None = None,
    add_score: bool = False,
) -> list[str]:
    print(f"[merge] processing shard: {shard_path.name}")
    tensors = st.load_file(str(shard_path))
    out_tensors: dict[str, torch.Tensor] = {}
    for name, w in tensors.items():
        if name in lora_deltas:
            out_tensors[name] = w + lora_deltas[name].to(w.device)
            print(f"  applied delta to {name}")
        else:
            out_tensors[name] = w

    if add_score and score_weight is not None:
        out_tensors["score.weight"] = score_weight
        print(f"  added score.weight {score_weight.shape}")

    st.save_file(out_tensors, str(output_path))
    return list(out_tensors.keys())


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    adapter = load_adapter(ADAPTER)
    lora_deltas = build_lora_delta_map(adapter)

    score_weight = adapter.get("base_model.model.score.weight")
    if score_weight is None:
        raise RuntimeError("score.weight not found in adapter!")
    print(f"[merge] score.weight shape: {score_weight.shape}, dtype: {score_weight.dtype}")

    index_file = BASE_MODEL / "model.safetensors.index.json"
    with open(index_file) as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]

    shard_files = sorted(set(weight_map.values()))
    new_weight_map: dict[str, str] = {}

    for i, shard_fname in enumerate(shard_files):
        src = BASE_MODEL / shard_fname
        dst = OUTPUT / shard_fname
        add_score_to_this = (i == len(shard_files) - 1)
        written_keys = process_shard(src, lora_deltas, dst, score_weight, add_score=add_score_to_this)
        for k in written_keys:
            new_weight_map[k] = shard_fname

    new_index = {"metadata": index.get("metadata", {}), "weight_map": new_weight_map}
    with open(OUTPUT / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)
    print(f"[merge] wrote model.safetensors.index.json, {len(new_weight_map)} weights")

    tokenizer_files = [
        "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
        "added_tokens.json", "special_tokens_map.json", "chat_template.jinja",
        "generation_config.json",
    ]
    for fname in tokenizer_files:
        src = BASE_MODEL / fname
        if src.exists():
            shutil.copy(src, OUTPUT / fname)
            print(f"[merge] copied {fname}")

    with open(BASE_MODEL / "config.json") as f:
        config = json.load(f)
    config["architectures"] = ["Qwen3MoeForSequenceClassification"]
    config["num_labels"] = 1
    config["classifier_dropout"] = 0.0
    with open(OUTPUT / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("[merge] wrote config.json with Qwen3MoeForSequenceClassification")

    last_shard = OUTPUT / shard_files[-1]
    check = st.load_file(str(last_shard))
    if "score.weight" in check:
        print(f"[merge] ✓ score.weight verified in {shard_files[-1]}: {check['score.weight'].shape}")
    else:
        print("[merge] ✗ score.weight NOT found in last shard!")

    print(f"\n[merge] Done → {OUTPUT}")


if __name__ == "__main__":
    main()
