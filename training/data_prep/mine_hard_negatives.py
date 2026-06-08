#!/usr/bin/env python3
"""
ChatJoy RM Hard Negative Mining
================================
用训好的 RM 对全量训练集打分，找出 chosen_reward < rejected_reward 的错误对。

单卡用法（v3 及之后，score head 在 adapter 里，无需额外参数）:
    CUDA_VISIBLE_DEVICES=4 python mine_hard_negatives.py \\
        --ckpt_hf <adapter_dir> \\
        --output hard_negatives_v3.jsonl

v2 用法（需先用 recover_score_head.py 恢复 score head）:
    CUDA_VISIBLE_DEVICES=4 python mine_hard_negatives.py \\
        --ckpt_hf <v2_adapter_dir> \\
        --score_head <v2_adapter_dir>/score_head_recovered.pt \\
        --output hard_negatives_v2.jsonl

双卡并行（速度翻倍，再用 merge 子命令合并）:
    CUDA_VISIBLE_DEVICES=4 python mine_hard_negatives.py \\
        --ckpt_hf <adapter> --score_head <head.pt> \\
        --output hard_negatives_shard0.jsonl \\
        --num_shards 2 --shard_index 0 &

    CUDA_VISIBLE_DEVICES=6 python mine_hard_negatives.py \\
        --ckpt_hf <adapter> --score_head <head.pt> \\
        --output hard_negatives_shard1.jsonl \\
        --num_shards 2 --shard_index 1 --device cuda:0 &

    wait && python mine_hard_negatives.py merge \\
        hard_negatives_shard0.jsonl hard_negatives_shard1.jsonl \\
        --output hard_negatives_v2.jsonl

score_head 文件说明:
    PEFT adapter 只保存 lora_* 权重，score.weight (frozen) 不在其中。
    v3 起已修复（modules_to_save）：score head 在 adapter 里，无需额外文件。
    v2 需要用 recover_score_head.py 先恢复。

    提取方式:
    A. 从 DS checkpoint (v1, global_step100): 见 extract_score_head() 函数
    B. 从 v2: 用 recover_score_head.py（推荐）
    C. 从 v3+: adapter 里直接有 modules_to_save.score.weight，无需额外操作
"""

import argparse
import json
import sys
import os
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from transformers import AutoTokenizer

sys.path.insert(0, '/home/ps/hanjiaqi/OpenRLHF')

PRETRAIN = "/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged"
TRAIN_DATA = "/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_train_filtered.jsonl"
EVAL_DATA  = "/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/rm_eval.jsonl"


# ── Score head 提取工具 ──────────────────────────────────────────────────────

def extract_score_head(ds_ckpt_dir: str, output_pt: str):
    """从 DeepSpeed ZeRO-2 checkpoint 提取 score.weight，直接读 zip 里的 tensor，不加载 30GB。

    原理：PyTorch 的 .pt 文件是 zip 压缩包，pkl 里存 key→storageID 映射，
    实际 tensor 数据在 zip 里的 data/<storageID> 文件，可以单独提取。
    经验证 key 为 base_model.model.score.weight，storageID 为 112227（可自动扫描）。
    """
    import zipfile, re, struct
    import numpy as np

    model_state_path = os.path.join(ds_ckpt_dir, "mp_rank_00_model_states.pt")
    if not os.path.exists(model_state_path):
        raise FileNotFoundError(f"Not found: {model_state_path}")

    print(f"Scanning {model_state_path} for score.weight (no full load needed) ...", flush=True)
    zf = zipfile.ZipFile(model_state_path)
    pkl_file = next(n for n in zf.namelist() if n.endswith('.pkl'))

    with zf.open(pkl_file) as f:
        raw = f.read()

    # 找 score.weight 的 storageID（格式：...score.weight\rXXXX h\x05((h\x06h\x07 X\x06...<id>...）
    # pkl 里 storage 引用格式: torch.BFloat16Storage, key="<storage_id>", device="cuda:0"
    # key 紧跟在 score.weight 之后的 X\x06\x00\x00\x00 pattern
    score_match = re.search(rb'base_model\.model\.score\.weight[^\x58]+\x58([\x01-\x7f])\x00\x00\x00([0-9]+)', raw)
    if not score_match:
        raise RuntimeError("Could not locate score.weight storage id in pkl. "
                           "Checkpoint may use a different key format.")

    storage_id = score_match.group(2).decode('ascii')
    print(f"  Found storage_id: {storage_id}", flush=True)

    # tensor 数据文件路径（PyTorch zip 格式：<prefix>/data/<storageID>）
    prefix = pkl_file.rsplit('/', 1)[0]  # e.g. "mp_rank_00_model_states"
    data_file = f"{prefix}/data/{storage_id}"
    if data_file not in zf.namelist():
        raise RuntimeError(f"Storage data file not found in zip: {data_file}")

    with zf.open(data_file) as f:
        raw_bytes = f.read()

    print(f"  Raw tensor bytes: {len(raw_bytes)}", flush=True)
    # BFloat16: 2 bytes per element，hidden_size=7168 → shape [1, 7168]
    n_elements = len(raw_bytes) // 2
    arr = np.frombuffer(raw_bytes, dtype=np.uint16).copy()
    weight = torch.from_numpy(arr).view(torch.bfloat16).reshape(1, -1)
    print(f"  score.weight shape: {weight.shape}, dtype: {weight.dtype}", flush=True)

    torch.save({'score.weight': weight}, output_pt)
    print(f"Saved → {output_pt}", flush=True)


# ── 模型加载 ─────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(ckpt_hf: str, score_head_pt: str = None, device: str = "cuda:0"):
    from openrlhf.models import get_llm_for_sequence_regression
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model (4-bit) from {PRETRAIN} ...", flush=True)
    # lora_rank=0: 加载纯 base RewardModel，不初始化 LoRA
    model = get_llm_for_sequence_regression(
        PRETRAIN,
        "reward",
        param_dtype="bf16",
        load_in_4bit=True,
        lora_rank=0,                        # 不初始化 LoRA，后面从 ckpt 加载
        attn_implementation="flash_attention_2",
        value_head_prefix="score",
        init_value_head=True,               # 先用默认 init，后面会被 patch 或 adapter 覆盖
        ds_config=None,
        device_map={"": device},
    )

    print(f"Loading LoRA adapter from {ckpt_hf} ...", flush=True)
    model = PeftModel.from_pretrained(model, ckpt_hf)
    # modules_to_save 修复后，score head 也在 adapter 里；否则需要外部 .pt

    # 覆盖 score head（如果提供了单独文件）
    if score_head_pt and os.path.exists(score_head_pt):
        print(f"Patching score head from {score_head_pt} ...", flush=True)
        sd = torch.load(score_head_pt, map_location='cpu', weights_only=True)
        score_key = next((k for k in sd if 'score' in k), None)
        if score_key:
            w = sd[score_key].to(torch.bfloat16).to(device)
            # 处理 PeftModel 包装层级
            base = model.base_model if hasattr(model, 'base_model') else model
            score_head = getattr(base, 'score', None) or getattr(base.model, 'score', None)
            if score_head is not None:
                with torch.no_grad():
                    score_head.weight.data.copy_(w)
                print(f"  Patched score.weight: {w.shape}", flush=True)
            else:
                print("  WARNING: Could not find score head module to patch", flush=True)
        else:
            print(f"  WARNING: No score key in {score_head_pt}, keys: {list(sd.keys())}", flush=True)
    elif score_head_pt:
        print(f"WARNING: --score_head file not found: {score_head_pt}", flush=True)
        print("         Score head is randomly initialized — inference quality degraded.", flush=True)
        print("         Use gap_threshold <= -0.5 for reliable results.", flush=True)
    else:
        print("WARNING: No --score_head provided.", flush=True)
        print("         v2 fix (modules_to_save) should include it in adapter automatically.", flush=True)
        print("         If adapter was trained with old code, use -0.5 threshold.", flush=True)

    model.eval()
    return model, tokenizer


# ── 分词 ────────────────────────────────────────────────────────────────────

def tokenize_pair(tokenizer, prompt_msgs, chosen_msgs, rejected_msgs, max_len):
    """复现 OpenRLHF RewardDataset 的分词逻辑（apply_chat_template + prompt_key）。"""
    tmpl = tokenizer.apply_chat_template
    prompt_str   = tmpl(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_chosen  = tmpl(prompt_msgs + chosen_msgs,   tokenize=False)
    full_rejected = tmpl(prompt_msgs + rejected_msgs, tokenize=False)

    chosen_resp   = full_chosen[len(prompt_str):]
    rejected_resp = full_rejected[len(prompt_str):]

    c_enc = tokenizer(
        prompt_str + chosen_resp,
        truncation=True, max_length=max_len,
        return_tensors="pt", padding=False,
    )
    r_enc = tokenizer(
        prompt_str + rejected_resp,
        truncation=True, max_length=max_len,
        return_tensors="pt", padding=False,
    )
    return c_enc.input_ids, c_enc.attention_mask, r_enc.input_ids, r_enc.attention_mask


# ── 推理（支持 batch） ───────────────────────────────────────────────────────

@torch.no_grad()
def score_batch(model, ids_list, masks_list, device):
    """对一组序列批量打分，返回分数列表。用 left-padding 对齐。"""
    max_len = max(ids.size(1) for ids in ids_list)
    padded_ids   = []
    padded_masks = []
    for ids, mask in zip(ids_list, masks_list):
        pad_len = max_len - ids.size(1)
        if pad_len > 0:
            pad_id = torch.zeros(1, pad_len, dtype=ids.dtype)
            ids  = torch.cat([pad_id, ids], dim=1)
            mask = torch.cat([torch.zeros(1, pad_len, dtype=mask.dtype), mask], dim=1)
        padded_ids.append(ids)
        padded_masks.append(mask)

    ids_t   = torch.cat(padded_ids,   dim=0).to(device)
    masks_t = torch.cat(padded_masks, dim=0).to(device)
    scores  = model(ids_t, masks_t)   # [batch]
    return scores.tolist()


# ── 主流程 ───────────────────────────────────────────────────────────────────

def print_stats(bad_cases, total):
    print(f"\n{'='*60}", flush=True)
    print(f"  Bad cases: {len(bad_cases)} / {total} "
          f"({100*len(bad_cases)/total:.1f}%)", flush=True)

    if bad_cases:
        gaps = [r["gap"] for r in bad_cases]
        gs = sorted(gaps)
        n  = len(gs)
        print(f"\n  Gap distribution:", flush=True)
        print(f"    worst:  {gs[0]:.4f}", flush=True)
        print(f"    p10:    {gs[n//10]:.4f}", flush=True)
        print(f"    median: {gs[n//2]:.4f}", flush=True)
        print(f"    mean:   {sum(gs)/n:.4f}", flush=True)

        types = Counter(r.get("extra_info", {}).get("rejected_type", "unknown") for r in bad_cases)
        print(f"\n  By rejected_type:", flush=True)
        for t, cnt in sorted(types.items(), key=lambda x: -x[1]):
            print(f"    {t:30s}: {cnt:4d}  ({100*cnt/len(bad_cases):.1f}%)", flush=True)

        langs = Counter(r.get("extra_info", {}).get("language", "unknown") for r in bad_cases)
        print(f"\n  Top 10 languages:", flush=True)
        for lang, cnt in sorted(langs.items(), key=lambda x: -x[1])[:10]:
            print(f"    {lang:10s}: {cnt:4d}", flush=True)


def append_records(records, output_path):
    if not records:
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()


def run_mining(args):
    device = args.device
    model, tokenizer = load_model_and_tokenizer(args.ckpt_hf, args.score_head, device)

    all_data = [json.loads(l) for l in open(args.train_data) if l.strip()]
    if args.filter_rejected_type:
        allowed_types = {t.strip() for t in args.filter_rejected_type.split(",") if t.strip()}
        all_data = [
            d for d in all_data
            if d.get("extra_info", {}).get("rejected_type", "unknown") in allowed_types
        ]
        print(f"Applied rejected_type filter: {sorted(allowed_types)}", flush=True)

    # Sharding：每个 shard 处理自己的子集（按 index 取模）
    if args.num_shards > 1:
        data = [d for i, d in enumerate(all_data) if i % args.num_shards == args.shard_index]
        global_indices = [i for i in range(len(all_data)) if i % args.num_shards == args.shard_index]
        print(f"\n[Shard {args.shard_index}/{args.num_shards}] "
              f"{len(data)} pairs (global idx {global_indices[0]}..{global_indices[-1]})", flush=True)
    else:
        data = all_data
        global_indices = list(range(len(all_data)))

    print(f"Scoring {len(data)} pairs  threshold={args.gap_threshold}  "
          f"device={device}  batch_size={args.batch_size}", flush=True)

    bad_cases, errors = [], 0
    batch_buf = []  # (global_idx, d, c_ids, c_mask, r_ids, r_mask)
    pending_writes = []
    write_every = max(100, args.batch_size * 50)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    open(args.output, "w").close()

    def flush_batch():
        nonlocal errors
        if not batch_buf:
            return
        try:
            ids_list   = [x[2] for x in batch_buf] + [x[4] for x in batch_buf]
            masks_list = [x[3] for x in batch_buf] + [x[5] for x in batch_buf]
            scores = score_batch(model, ids_list, masks_list, device)
            bs = len(batch_buf)
            c_scores = scores[:bs]
            r_scores = scores[bs:]
            for (gidx, d, _, _, _, _), c_s, r_s in zip(batch_buf, c_scores, r_scores):
                gap = c_s - r_s
                if gap <= args.gap_threshold:
                    record = {
                        "idx": gidx,
                        "gap": round(gap, 4),
                        "chosen_reward":   round(c_s, 4),
                        "rejected_reward": round(r_s, 4),
                        "prompt":   d["prompt"],
                        "chosen":   d["chosen"],
                        "rejected": d["rejected"],
                    }
                    if "extra_info" in d:
                        record["extra_info"] = d["extra_info"]
                    bad_cases.append(record)
                    pending_writes.append(record)
        except Exception as e:
            errors += len(batch_buf)
            if errors <= 10:
                print(f"  Batch error: {e}", flush=True)
        batch_buf.clear()

        if len(pending_writes) >= write_every:
            pending_writes.sort(key=lambda x: x["gap"])
            append_records(pending_writes, args.output)
            print(f"  appended {len(pending_writes)} records -> {args.output}", flush=True)
            pending_writes.clear()

    for local_i, (gidx, d) in enumerate(zip(global_indices, data)):
        if local_i % 500 == 0:
            print(f"  [{local_i:5d}/{len(data)}]  bad={len(bad_cases)}  errors={errors}", flush=True)
        try:
            c_ids, c_mask, r_ids, r_mask = tokenize_pair(
                tokenizer, d["prompt"], d["chosen"], d["rejected"], args.max_len,
            )
            batch_buf.append((gidx, d, c_ids, c_mask, r_ids, r_mask))
            if len(batch_buf) >= args.batch_size:
                flush_batch()
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Tokenize error idx={gidx}: {e}", flush=True)

    flush_batch()  # 处理最后一批
    if pending_writes:
        pending_writes.sort(key=lambda x: x["gap"])
        append_records(pending_writes, args.output)
        print(f"  appended {len(pending_writes)} records -> {args.output}", flush=True)

    print(f"\n  Output: {args.output}", flush=True)
    print_stats(bad_cases, len(data))


# ── Shard 合并 ───────────────────────────────────────────────────────────────

def run_merge(shard_files, output, threshold):
    all_records = []
    for sf in shard_files:
        recs = [json.loads(l) for l in open(sf) if l.strip()]
        print(f"  {sf}: {len(recs)} records", flush=True)
        all_records.extend(recs)

    # 按 gap 排序，去重（同 idx 取 gap 最小的）
    seen = {}
    for r in all_records:
        idx = r["idx"]
        if idx not in seen or r["gap"] < seen[idx]["gap"]:
            seen[idx] = r

    merged = sorted(seen.values(), key=lambda x: x["gap"])
    merged = [r for r in merged if r["gap"] <= threshold]

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nMerged {len(all_records)} → {len(merged)} records (threshold={threshold})", flush=True)
    print(f"Output: {output}", flush=True)
    print_stats(merged, len(merged))


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ChatJoy RM Hard Negative Mining")
    sub = parser.add_subparsers(dest="cmd")

    # 提取 score head（从 DS checkpoint）
    ext = sub.add_parser("extract_score_head", help="Extract score.weight from DS checkpoint")
    ext.add_argument("ds_ckpt_dir")
    ext.add_argument("output_pt")

    # shard 合并
    mg = sub.add_parser("merge", help="Merge shard outputs into final bad case pool")
    mg.add_argument("shard_files", nargs="+", help="Shard jsonl files to merge")
    mg.add_argument("--output",       required=True)
    mg.add_argument("--gap_threshold", type=float, default=-0.3)

    # 主推理
    parser.add_argument("--ckpt_hf",        required=False)
    parser.add_argument("--score_head",     default=None,
                        help="score_head.pt (v2: recovered; v3+: not needed)")
    parser.add_argument("--train_data",     default=TRAIN_DATA)
    parser.add_argument("--output",         default="/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm/hard_negatives.jsonl")
    parser.add_argument("--gap_threshold",  type=float, default=-0.3)
    parser.add_argument("--filter_rejected_type", default=None,
                        help="Comma-separated rejected_type filter, e.g. length_mismatch_long")
    parser.add_argument("--max_len",        type=int, default=3072)
    parser.add_argument("--device",         default="cuda:0")
    parser.add_argument("--batch_size",     type=int, default=2,
                        help="Pairs per batch (default 2; increase for speedup if VRAM allows)")
    parser.add_argument("--num_shards",     type=int, default=1,
                        help="Total number of parallel shards")
    parser.add_argument("--shard_index",    type=int, default=0,
                        help="This shard's index (0-based)")

    args = parser.parse_args()

    if args.cmd == "extract_score_head":
        extract_score_head(args.ds_ckpt_dir, args.output_pt)
        return

    if args.cmd == "merge":
        run_merge(args.shard_files, args.output, args.gap_threshold)
        return

    if not args.ckpt_hf:
        parser.error("--ckpt_hf is required for mining")

    run_mining(args)


if __name__ == "__main__":
    main()
