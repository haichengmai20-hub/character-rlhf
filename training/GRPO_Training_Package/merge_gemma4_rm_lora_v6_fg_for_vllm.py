#!/usr/bin/env python3
"""Merge a Gemma4 SFT-base RM v6 LoRA adapter into a vLLM /classify checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import safetensors.torch as st
import torch


BASE = Path("/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b")
DEFAULT_BASE_MODEL = BASE / "sft2_ddp_lora_3gpu_567_cut3072_merged"


def load_adapter(adapter_path: Path) -> dict[str, torch.Tensor]:
    adapter_file = adapter_path / "adapter_model.safetensors"
    print(f"[merge] loading adapter: {adapter_file}", flush=True)
    tensors = st.load_file(str(adapter_file))
    print(f"[merge] adapter tensors: {len(tensors)}", flush=True)
    return tensors


def adapter_key_to_base_key(adapter_key: str) -> str:
    return adapter_key.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")


def build_lora_delta_map(adapter: dict[str, torch.Tensor], lora_alpha: int, lora_rank: int) -> dict[str, torch.Tensor]:
    scaling = lora_alpha / lora_rank
    deltas: dict[str, torch.Tensor] = {}
    a_keys = sorted(k for k in adapter if ".lora_A.weight" in k)
    print(f"[merge] lora targets: {len(a_keys)} scaling={scaling}", flush=True)

    for a_key in a_keys:
        b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
        if b_key not in adapter:
            print(f"[merge][warn] missing lora_B for {a_key}", flush=True)
            continue
        a = adapter[a_key]
        b = adapter[b_key]
        delta = (b.float() @ a.float()) * scaling
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
    if add_score and score_weight is not None:
        out_tensors["score.weight"] = score_weight
    st.save_file(out_tensors, str(output_path))
    print(f"[merge] wrote {output_path.name}, applied={applied}", flush=True)
    return list(out_tensors.keys())


def copy_tokenizer_and_processor_files(base_model: Path, output: Path) -> None:
    for name in ["tokenizer_config.json", "tokenizer.json", "chat_template.jinja", "generation_config.json", "processor_config.json"]:
        src = base_model / name
        if src.exists():
            shutil.copy(src, output / name)


def write_sequence_classification_config(base_model: Path, output: Path) -> None:
    with open(base_model / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    config["architectures"] = ["Gemma4ForSequenceClassification"]
    config["num_labels"] = 1
    config["classifier_dropout"] = 0.0
    text_config = config.setdefault("text_config", {})
    text_config["num_labels"] = 1
    text_config["classifier_dropout"] = 0.0
    with open(output / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=64)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    adapter = load_adapter(args.adapter)
    lora_deltas = build_lora_delta_map(adapter, lora_alpha=args.lora_alpha, lora_rank=args.lora_rank)
    score_weight = adapter.get("base_model.model.score.weight")
    if score_weight is None:
        raise RuntimeError("base_model.model.score.weight not found in adapter")

    with open(args.base_model / "model.safetensors.index.json", encoding="utf-8") as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    new_weight_map: dict[str, str] = {}
    for idx, shard_name in enumerate(shard_files):
        written_keys = process_shard(
            args.base_model / shard_name,
            lora_deltas,
            args.output / shard_name,
            score_weight,
            add_score=(idx == len(shard_files) - 1),
        )
        for key in written_keys:
            new_weight_map[key] = shard_name

    with open(args.output / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": index.get("metadata", {}), "weight_map": new_weight_map}, f, indent=2)
    copy_tokenizer_and_processor_files(args.base_model, args.output)
    write_sequence_classification_config(args.base_model, args.output)
    print(f"[merge] done: {args.output}", flush=True)


if __name__ == "__main__":
    main()

