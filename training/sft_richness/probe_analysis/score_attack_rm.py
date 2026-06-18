#!/usr/bin/env python3
"""
Score an attack probe JSONL with the v6.6 ChatJoy reward model.

Usage:
  CUDA_VISIBLE_DEVICES=4,5,6,7 python3 score_attack_rm.py \
      --in step0_attack_t0.9.jsonl \
      --out step0_attack_rm.jsonl \
      --label step0
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


DEFAULT_RM_PATH = "/root/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_v6.6_fg_7gpu_merged_vllm"


def probs0(req_out) -> float:
    outputs = req_out.outputs
    probs = getattr(outputs, "probs", None)
    if probs is not None:
        return float(list(probs)[0])
    return float(np.asarray(outputs.data).ravel()[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="model")
    parser.add_argument("--rm", default=DEFAULT_RM_PATH)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM

    records = [json.loads(line) for line in open(args.inp, encoding="utf-8") if line.strip()]
    print(f"[load] {len(records)} records from {args.inp}")

    tokenizer = AutoTokenizer.from_pretrained(args.rm, trust_remote_code=True)

    def render(record: dict) -> str:
        messages = []
        if record.get("system"):
            messages.append({"role": "system", "content": record["system"]})
        if record.get("user"):
            messages.append({"role": "user", "content": record["user"]})
        if record.get("response"):
            messages.append({"role": "assistant", "content": record["response"]})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    prompts = [render(record) for record in records]

    t0 = time.time()
    llm = LLM(
        model=args.rm,
        runner="pooling",
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_len,
        limit_mm_per_prompt={"image": 0, "video": 0},
        trust_remote_code=True,
        enforce_eager=True,
    )
    print(f"[load] RM loaded in {time.time() - t0:.1f}s")

    t1 = time.time()
    outs = llm.classify(prompts)
    print(f"[score] {len(prompts)} records in {time.time() - t1:.1f}s")

    with open(args.out, "w", encoding="utf-8") as file:
        for record, rm_out in zip(records, outs):
            record["probs_0"] = probs0(rm_out)
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    attack_scores = [row["probs_0"] for row in records if "A" in row.get("case_id", "")]
    clean_scores = [row["probs_0"] for row in records if "A" not in row.get("case_id", "")]

    print(f"\n=== RM Summary [{args.label}] ===")
    if attack_scores:
        arr = np.asarray(attack_scores)
        print(f"  attack-split  n={len(arr):3d}  mean={arr.mean():.4f}  min={arr.min():.3f}  max={arr.max():.3f}")
    if clean_scores:
        arr = np.asarray(clean_scores)
        print(f"  clean-split   n={len(arr):3d}  mean={arr.mean():.4f}  min={arr.min():.3f}  max={arr.max():.3f}")
    print("  anchors: step0@t0.7 attack-split approx 0.293, step0@t1.0 approx 0.254, base approx 0.194")


if __name__ == "__main__":
    main()
