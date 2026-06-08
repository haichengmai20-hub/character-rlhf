#!/usr/bin/env python3
"""
Merge the Gemma4 SFT-base RM v5 LoRA adapter into the Gemma4 SFT base.

Output is a vLLM /classify-compatible HF checkpoint:
  saves/gemma4-26b-a4b/chatjoy_rm_gemma4_sft_v5_merged_vllm/
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import safetensors.torch as st
import torch


BASE = Path("/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b")
BASE_MODEL = BASE / "sft2_ddp_lora_3gpu_567_cut3072_merged"
ADAPTER = BASE / "chatjoy_rm_gemma4_sft_v5_20260509_101133"
OUTPUT = BASE / "chatjoy_rm_gemma4_sft_v5_merged_vllm"

LORA_ALPHA = 128
LORA_RANK = 64
SCALING = LORA_ALPHA / LORA_RANK


def load_adapter(adapter_path: Path) -> dict[str, torch.Tensor]:
    adapter_file = adapter_path / "adapter_model.safetensors"
    print(f"[merge] loading adapter: {adapter_file}", flush=True)
    tensors = st.load_file(str(adapter_file))
    print(f"[merge] adapter tensors: {len(tensors)}", flush=True)
    return tensors


def adapter_key_to_base_key(adapter_key: str) -> str:
    # Adapter:
    #   base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight
    # Base:
    #   model.language_model.layers.0.self_attn.q_proj.weight
    return adapter_key.replace("base_model.model.", "").replace(
        ".lora_A.weight", ".weight"
    )


def build_lora_delta_map(adapter: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    deltas: dict[str, torch.Tensor] = {}
    a_keys = sorted(k for k in adapter if ".lora_A.weight" in k)
    print(f"[merge] lora targets: {len(a_keys)}", flush=True)

    for a_key in a_keys:
        b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
        if b_key not in adapter:
            print(f"[merge][warn] missing lora_B for {a_key}", flush=True)
            continue

        a = adapter[a_key]
        b = adapter[b_key]
        delta = (b.float() @ a.float()) * SCALING
        deltas[adapter_key_to_base_key(a_key)] = delta.to(a.dtype)

    return deltas


def process_shard(
    shard_path: Path,
    lora_deltas: dict[str, torch.Tensor],
    output_path: Path,
    score_weight: torch.Tensor | None,
    add_score: bool,
) -> list[str]:
    print(f"[merge] processing shard: {shard_path.name}", flush=True)
    tensors = st.load_file(str(shard_path))
    out_tensors: dict[str, torch.Tensor] = {}
    applied = 0

    for name, weight in tensors.items():
        delta = lora_deltas.get(name)
        if delta is None:
            out_tensors[name] = weight
            continue

        out_tensors[name] = weight + delta.to(device=weight.device, dtype=weight.dtype)
        applied += 1
        print(f"  applied delta: {name}", flush=True)

    if add_score and score_weight is not None:
        out_tensors["score.weight"] = score_weight
        print(f"  added score.weight {tuple(score_weight.shape)}", flush=True)

    st.save_file(out_tensors, str(output_path))
    print(f"[merge] wrote {output_path.name}, applied={applied}", flush=True)
    return list(out_tensors.keys())


def copy_tokenizer_and_processor_files() -> None:
    for name in [
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
    ]:
        src = BASE_MODEL / name
        if src.exists():
            shutil.copy(src, OUTPUT / name)
            print(f"[merge] copied {name}", flush=True)


def write_sequence_classification_config() -> None:
    with open(BASE_MODEL / "config.json") as f:
        config = json.load(f)

    config["architectures"] = ["Gemma4ForSequenceClassification"]
    config["num_labels"] = 1
    config["classifier_dropout"] = 0.0

    text_config = config.setdefault("text_config", {})
    text_config["num_labels"] = 1
    text_config["classifier_dropout"] = 0.0

    with open(OUTPUT / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("[merge] wrote config.json with Gemma4ForSequenceClassification", flush=True)


def main() -> None:
    if not BASE_MODEL.exists():
        raise FileNotFoundError(BASE_MODEL)
    if not ADAPTER.exists():
        raise FileNotFoundError(ADAPTER)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    adapter = load_adapter(ADAPTER)
    lora_deltas = build_lora_delta_map(adapter)

    score_weight = adapter.get("base_model.model.score.weight")
    if score_weight is None:
        raise RuntimeError("base_model.model.score.weight not found in adapter")
    print(
        f"[merge] score.weight shape={tuple(score_weight.shape)} dtype={score_weight.dtype}",
        flush=True,
    )

    with open(BASE_MODEL / "model.safetensors.index.json") as f:
        index = json.load(f)

    shard_files = sorted(set(index["weight_map"].values()))
    new_weight_map: dict[str, str] = {}

    for idx, shard_name in enumerate(shard_files):
        written_keys = process_shard(
            BASE_MODEL / shard_name,
            lora_deltas,
            OUTPUT / shard_name,
            score_weight,
            add_score=(idx == len(shard_files) - 1),
        )
        for key in written_keys:
            new_weight_map[key] = shard_name

    with open(OUTPUT / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": index.get("metadata", {}), "weight_map": new_weight_map}, f, indent=2)
    print(f"[merge] wrote model.safetensors.index.json, weights={len(new_weight_map)}", flush=True)

    copy_tokenizer_and_processor_files()
    write_sequence_classification_config()

    check = st.load_file(str(OUTPUT / shard_files[-1]))
    if "score.weight" not in check:
        raise RuntimeError("score.weight missing from merged output")
    print(f"[merge] verified score.weight in {shard_files[-1]}", flush=True)
    print(f"[merge] done: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
