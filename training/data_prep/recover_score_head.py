#!/usr/bin/env python3
"""
Score Head Recovery for RM v2
==============================
v2 训练时 model.py 未修复 modules_to_save，score.weight 未被保存。
此脚本在固定的 base + v2 LoRA 特征上重新学习 score.weight。

原理：
  LoRA 已学好表征（83.7% eval acc 证实）。score.weight 是 2816→1
  的线性投影。在固定特征上做 Margin Ranking Loss 是凸优化，收敛快。
  关键：transformer forward 用 no_grad + hidden state detach，
  梯度只流向 score.weight，不需要 backprop 穿整个模型。

两阶段：
  Phase 1  (1-2h)  批量 inference，把所有序列的 EOS hidden state
                   存到 CPU 内存（~170MB）。
  Phase 2  (<5min) 只在 2816 维特征上跑 AdamW，几个 epoch 收敛。

用法:
  CUDA_VISIBLE_DEVICES=4 python recover_score_head.py \\
      --ckpt_hf /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_rm_v2_filtered \\
      --output  /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_rm_v2_filtered/score_head_recovered.pt \\
      --validate_on /home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_eval.jsonl
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer

sys.path.insert(0, '/home/ps/hanjiaqi/OpenRLHF')

PRETRAIN   = "/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged"
TRAIN_DATA = "/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_train_filtered.jsonl"


# ── 模型加载 ─────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(ckpt_hf: str, device: str):
    from openrlhf.models import get_llm_for_sequence_regression
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model (4-bit) ...", flush=True)
    model = get_llm_for_sequence_regression(
        PRETRAIN, "reward",
        param_dtype="bf16",
        load_in_4bit=True,
        lora_rank=0,
        attn_implementation="flash_attention_2",
        value_head_prefix="score",
        init_value_head=True,
        ds_config=None,
        device_map={"": device},
    )

    print(f"Loading LoRA adapter from {ckpt_hf} ...", flush=True)
    model = PeftModel.from_pretrained(model, ckpt_hf)
    model.eval()

    # 冻结所有参数（含 score head）
    for p in model.parameters():
        p.requires_grad_(False)

    return model, tokenizer


# ── 分词 ─────────────────────────────────────────────────────────────────────

def tokenize_single(tokenizer, prompt_msgs, response_msgs, max_len):
    tmpl = tokenizer.apply_chat_template
    prompt_str   = tmpl(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_str     = tmpl(prompt_msgs + response_msgs, tokenize=False)
    resp_str     = full_str[len(prompt_str):]
    enc = tokenizer(
        prompt_str + resp_str,
        truncation=True, max_length=max_len,
        return_tensors="pt", padding=False,
    )
    return enc.input_ids, enc.attention_mask


# ── EOS hidden state 提取 ────────────────────────────────────────────────────

@torch.no_grad()
def extract_eos_hidden(model, input_ids, attention_mask, device):
    """返回 EOS 位置的 hidden state，shape [hidden_size]"""
    input_ids   = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # return_output=True 获得 last_hidden_state
    _, outputs = model(input_ids, attention_mask, return_output=True)
    last_hidden = outputs["last_hidden_state"]  # [1, seq_len, hidden]

    eos_idx = (attention_mask.size(1) - 1
               - attention_mask.long().fliplr().argmax(dim=1, keepdim=True))  # [1,1]
    hidden = last_hidden.gather(
        1,
        eos_idx.unsqueeze(-1).expand(-1, -1, last_hidden.size(-1))
    ).squeeze(0).squeeze(0)  # [hidden_size]

    return hidden.to(torch.float32).cpu()


# ── Phase 1：批量提取特征 ────────────────────────────────────────────────────

def phase1_extract(
    model,
    tokenizer,
    data,
    max_len,
    device,
    log_every=200,
    chosen_feats=None,
    rejected_feats=None,
    save_path=None,
    indices=None,
    total_size=None,
    save_every=50,
):
    chosen_feats   = list(chosen_feats or [])
    rejected_feats = list(rejected_feats or [])
    if len(chosen_feats) != len(rejected_feats):
        raise ValueError("chosen_feats and rejected_feats must have the same length.")
    errors = 0
    start_idx = len(chosen_feats)

    if start_idx > len(data):
        raise ValueError(f"Existing cache length {start_idx} exceeds shard size {len(data)}.")

    print(
        f"\n[Phase 1] Extracting EOS hidden states from {len(data)} pairs "
        f"(resume from {start_idx}) ...",
        flush=True,
    )
    for i, d in enumerate(data[start_idx:], start=start_idx):
        if i % log_every == 0:
            print(f"  [{i:5d}/{len(data)}]  errors={errors}", flush=True)
        try:
            c_ids, c_mask = tokenize_single(tokenizer, d["prompt"], d["chosen"],   max_len)
            r_ids, r_mask = tokenize_single(tokenizer, d["prompt"], d["rejected"], max_len)

            c_feat = extract_eos_hidden(model, c_ids, c_mask, device)
            r_feat = extract_eos_hidden(model, r_ids, r_mask, device)

            chosen_feats.append(c_feat)
            rejected_feats.append(r_feat)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error idx={i}: {e}", flush=True)
            chosen_feats.append(None)
            rejected_feats.append(None)

        if save_path and ((i + 1) % save_every == 0 or i + 1 == len(data)):
            save_feature_cache(
                save_path,
                chosen_feats,
                rejected_feats,
                indices=indices,
                total_size=total_size,
            )
            print(f"  autosaved cache: {i + 1}/{len(data)} -> {save_path}", flush=True)

    print(f"[Phase 1] Done. Errors: {errors}/{len(data)}", flush=True)
    return chosen_feats, rejected_feats


def save_feature_cache(cache_path: str, chosen_feats, rejected_feats, indices=None, total_size=None):
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"chosen": chosen_feats, "rejected": rejected_feats}
    if indices is not None:
        payload["indices"] = indices
    if total_size is not None:
        payload["total_size"] = total_size
    torch.save(payload, cache_path)


def load_feature_cache(cache_path: str):
    return torch.load(cache_path, weights_only=True)


def merge_feature_caches(cache_paths: List[str], output_path: str):
    caches = [load_feature_cache(p) for p in cache_paths]
    if not caches:
        raise ValueError("No cache files provided.")

    total_sizes = {cache.get("total_size") for cache in caches}
    if len(total_sizes) != 1 or None in total_sizes:
        raise ValueError("All shard caches must contain the same non-null total_size.")

    total_size = total_sizes.pop()
    chosen = [None] * total_size
    rejected = [None] * total_size

    for cache_path, cache in zip(cache_paths, caches):
        indices = cache.get("indices")
        if indices is None:
            raise ValueError(f"Cache {cache_path} has no shard indices.")
        if len(indices) != len(cache["chosen"]) or len(indices) != len(cache["rejected"]):
            raise ValueError(f"Cache {cache_path} has inconsistent shard lengths.")
        for idx, c_feat, r_feat in zip(indices, cache["chosen"], cache["rejected"]):
            if chosen[idx] is not None or rejected[idx] is not None:
                raise ValueError(f"Duplicate feature for index {idx} while merging caches.")
            chosen[idx] = c_feat
            rejected[idx] = r_feat

    save_feature_cache(output_path, chosen, rejected)
    valid = sum(c is not None and r is not None for c, r in zip(chosen, rejected))
    print(f"Merged {len(cache_paths)} caches → {output_path}", flush=True)
    print(f"  total={total_size}  valid={valid}", flush=True)


# ── Phase 2：优化 score head ─────────────────────────────────────────────────

def phase2_optimize(chosen_feats, rejected_feats, hidden_size=2816,
                    epochs=5, batch_size=256, lr=1e-3, margin=0.5):
    """在 CPU 上优化 2816→1 的线性层，margin ranking loss，收敛极快。"""

    # 过滤掉提取失败的
    valid = [(c, r) for c, r in zip(chosen_feats, rejected_feats)
             if c is not None and r is not None]
    print(f"\n[Phase 2] Optimizing score head on {len(valid)} valid pairs ...", flush=True)
    print(f"  hidden_size={hidden_size}, epochs={epochs}, batch_size={batch_size}, lr={lr}", flush=True)

    c_all = torch.stack([x[0] for x in valid])  # [N, H]
    r_all = torch.stack([x[1] for x in valid])  # [N, H]

    score_head = nn.Linear(hidden_size, 1, bias=False)
    nn.init.normal_(score_head.weight, mean=0.0, std=1.0 / (hidden_size + 1))

    optimizer = AdamW(score_head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MarginRankingLoss(margin=margin)
    target    = torch.ones(batch_size)

    n = len(valid)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        total_loss = 0.0
        correct = 0
        steps = 0

        for start in range(0, n, batch_size):
            idx  = perm[start:start + batch_size]
            c_b  = c_all[idx]
            r_b  = r_all[idx]
            bs   = c_b.size(0)
            tgt  = target[:bs]

            c_score = score_head(c_b).squeeze(-1)   # [bs]
            r_score = score_head(r_b).squeeze(-1)

            loss = criterion(c_score, r_score, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bs
            correct    += (c_score > r_score).sum().item()
            steps      += 1

        acc = correct / n
        avg_loss = total_loss / n
        print(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  acc={acc:.1%}", flush=True)

    return score_head


# ── 验证 ─────────────────────────────────────────────────────────────────────

def validate(score_head, chosen_feats, rejected_feats, label="train"):
    valid = [(c, r) for c, r in zip(chosen_feats, rejected_feats)
             if c is not None and r is not None]
    c_all = torch.stack([x[0] for x in valid])
    r_all = torch.stack([x[1] for x in valid])
    with torch.no_grad():
        c_s = score_head(c_all).squeeze(-1)
        r_s = score_head(r_all).squeeze(-1)
    acc = (c_s > r_s).float().mean().item()
    gaps = (c_s - r_s).tolist()
    mean_gap = sum(gaps) / len(gaps)
    print(f"  [{label}] acc={acc:.1%}  gap_mean={mean_gap:+.4f}  n={len(valid)}", flush=True)
    return acc


def infer_hidden_size(chosen_feats, rejected_feats):
    for feat in list(chosen_feats) + list(rejected_feats):
        if feat is not None:
            return int(feat.shape[-1])
    raise ValueError("No valid features found to infer hidden size.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Recover score head for RM v2")
    subparsers = parser.add_subparsers(dest="command")

    merge_parser = subparsers.add_parser("merge-caches", help="Merge sharded feature caches.")
    merge_parser.add_argument("caches", nargs="+", help="Shard cache .pt files")
    merge_parser.add_argument("--output", required=True, help="Merged cache output path")

    parser.add_argument("--ckpt_hf",      required=False,
                        help="Path to v2 HF adapter dir (chatjoy_rm_v2_filtered/)")
    parser.add_argument("--output",       required=False,
                        help="Output path for recovered score head .pt")
    parser.add_argument("--train_data",   default=TRAIN_DATA)
    parser.add_argument("--validate_on",  default=None,
                        help="Optional eval jsonl for accuracy report")
    parser.add_argument("--max_len",      type=int, default=2048,
                        help="Max token length for feature extraction (default 2048)")
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--epochs",       type=int, default=5)
    parser.add_argument("--batch_size",   type=int, default=256)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--margin",       type=float, default=0.5)
    parser.add_argument("--cache",        default=None,
                        help="Path to cache extracted features (avoid re-extraction)")
    parser.add_argument("--num_shards",   type=int, default=1,
                        help="Split Phase 1 extraction into N shards.")
    parser.add_argument("--shard_index",  type=int, default=0,
                        help="Current shard index for Phase 1 extraction.")
    parser.add_argument("--save_every",   type=int, default=50,
                        help="Autosave feature cache every N pairs during Phase 1.")
    args = parser.parse_args()

    if args.command == "merge-caches":
        merge_feature_caches(args.caches, args.output)
        return

    if not args.ckpt_hf or not args.output:
        parser.error("--ckpt_hf and --output are required unless using merge-caches.")

    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1.")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error("--shard_index must satisfy 0 <= shard_index < num_shards.")

    train_data = [json.loads(l) for l in open(args.train_data) if l.strip()]

    # Phase 1: feature extraction (可缓存)
    cache_path = args.cache or args.output.replace(".pt", "_feat_cache.pt")
    shard_indices = None
    if args.num_shards > 1:
        shard_indices = [i for i in range(len(train_data)) if i % args.num_shards == args.shard_index]

    if args.num_shards > 1:
        shard_data = [train_data[i] for i in shard_indices]
        existing_chosen = []
        existing_rejected = []
        if os.path.exists(cache_path):
            print(f"Loading cached shard features from {cache_path} ...", flush=True)
            cache = load_feature_cache(cache_path)
            cache_indices = cache.get("indices")
            cache_total_size = cache.get("total_size")
            if cache_indices is not None and cache_indices != shard_indices:
                raise ValueError("Existing shard cache indices do not match requested shard.")
            if cache_total_size is not None and cache_total_size != len(train_data):
                raise ValueError("Existing shard cache total_size does not match current train data.")
            existing_chosen = cache["chosen"]
            existing_rejected = cache["rejected"]
            print(f"Shard cache progress: {len(existing_chosen)}/{len(shard_data)} pairs", flush=True)

        if len(existing_chosen) < len(shard_data):
            model, tokenizer = load_model_and_tokenizer(args.ckpt_hf, args.device)
            print(
                f"Running Phase 1 shard {args.shard_index}/{args.num_shards} "
                f"with {len(shard_data)} pairs (global total {len(train_data)})",
                flush=True,
            )
            chosen_feats, rejected_feats = phase1_extract(
                model,
                tokenizer,
                shard_data,
                args.max_len,
                args.device,
                chosen_feats=existing_chosen,
                rejected_feats=existing_rejected,
                save_path=cache_path,
                indices=shard_indices,
                total_size=len(train_data),
                save_every=args.save_every,
            )
            print(f"Saving shard feature cache → {cache_path}", flush=True)
            save_feature_cache(
                cache_path,
                chosen_feats,
                rejected_feats,
                indices=shard_indices,
                total_size=len(train_data),
            )
        else:
            print(f"Shard cache already complete for shard {args.shard_index}/{args.num_shards}.", flush=True)
        print("Shard extraction finished. Merge shard caches before Phase 2 optimization.", flush=True)
        return

    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path} ...", flush=True)
        cache = load_feature_cache(cache_path)
        chosen_feats   = cache["chosen"]
        rejected_feats = cache["rejected"]
    else:
        model, tokenizer = load_model_and_tokenizer(args.ckpt_hf, args.device)
        chosen_feats, rejected_feats = phase1_extract(
            model, tokenizer, train_data, args.max_len, args.device,
            save_path=cache_path, save_every=args.save_every
        )
        print(f"Saving feature cache → {cache_path}", flush=True)
        save_feature_cache(cache_path, chosen_feats, rejected_feats)

    # Phase 2: optimize score head
    hidden_size = infer_hidden_size(chosen_feats, rejected_feats)
    print(f"Inferred feature hidden_size={hidden_size}", flush=True)
    score_head = phase2_optimize(
        chosen_feats, rejected_feats,
        hidden_size=hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
    )

    # 保存
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"score.weight": score_head.weight.data.clone()}, args.output)
    print(f"\nSaved recovered score head → {args.output}", flush=True)
    print(f"  shape: {score_head.weight.data.shape}, dtype: {score_head.weight.data.dtype}", flush=True)

    # 验证
    print("\n[Validation]", flush=True)
    validate(score_head, chosen_feats, rejected_feats, label="train")

    if args.validate_on:
        try:
            model, tokenizer = load_model_and_tokenizer(args.ckpt_hf, args.device)
            eval_data = [json.loads(l) for l in open(args.validate_on) if l.strip()]
            print(f"Extracting eval features ({len(eval_data)} pairs) ...", flush=True)
            eval_c, eval_r = phase1_extract(
                model, tokenizer, eval_data, args.max_len, args.device, log_every=100
            )
            validate(score_head, eval_c, eval_r, label="eval")
        except Exception as e:
            print(f"[Validation] Eval extraction failed: {e}", flush=True)
            print("Recovered score head has already been saved; rerun eval separately if needed.", flush=True)


if __name__ == "__main__":
    main()
