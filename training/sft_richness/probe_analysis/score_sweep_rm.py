#!/usr/bin/env python3
"""score_sweep_rm.py
用 RM 给 sweep 产出的 4 个新 jsonl 打分:
  step0_t0.8 / step0_t0.85 / step0_t0.9 / step100_t1.0

用法:
  /home/ubuntu/miniconda3/envs/gemma4-rl/bin/python3 score_sweep_rm.py
"""
import argparse, json, os, time
import numpy as np


def load_probe(path, limit=None):
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            records.append(json.loads(line))
    return records


def build_messages(rec):
    msgs = []
    if rec.get("system"):
        msgs.append({"role": "system", "content": rec["system"]})
    if rec.get("user"):
        msgs.append({"role": "user", "content": rec["user"]})
    if rec.get("response"):
        msgs.append({"role": "assistant", "content": rec["response"]})
    return msgs


def extract_candidates(req_out):
    o = req_out.outputs
    cand = {}
    probs = getattr(o, "probs", None)
    if probs is not None:
        pl = list(probs) if hasattr(probs, "__iter__") else [probs]
        for i, p in enumerate(pl):
            cand[f"probs_{i}"] = float(p)
    data = getattr(o, "data", None)
    if data is not None:
        try:
            arr = np.asarray(data, dtype=float).ravel()
            if arr.size == 1:
                cand["data_0"] = float(arr[0])
            else:
                cand["data_last"] = float(arr[-1])
                cand["data_mean"] = float(arr.mean())
        except Exception:
            pass
    sc = getattr(o, "score", None)
    if isinstance(sc, (int, float)):
        cand["score"] = float(sc)
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/root/outputs/v67_step0_vs_step100")
    ap.add_argument("--rm", default="/root/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_v6.6_fg_7gpu_merged_vllm")
    ap.add_argument("--out", default="/root/outputs/v67_step0_vs_step100/rm_scores_sweep.jsonl")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    from vllm import LLM
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.rm, trust_remote_code=True)

    groups = ["step0_t0.8", "step0_t0.85", "step0_t0.9", "step100_t1.0"]
    all_records = []
    for g in groups:
        path = os.path.join(a.data_dir, f"{g}.jsonl")
        recs = load_probe(path, a.limit)
        for r in recs:
            r["_group"] = g
        all_records.extend(recs)
        print(f"[load] {g}: {len(recs)} records")

    prompts = []
    for rec in all_records:
        msgs = build_messages(rec)
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompts.append(rendered)

    print(f"[score] total={len(prompts)} tp={a.tp} max_len={a.max_len}")

    t0 = time.time()
    llm = LLM(
        model=a.rm,
        runner="pooling",
        tensor_parallel_size=a.tp,
        gpu_memory_utilization=a.gpu_mem,
        max_model_len=a.max_len,
        limit_mm_per_prompt={"image": 0, "video": 0},
        trust_remote_code=True,
        enforce_eager=True,
    )
    print(f"[load] RM loaded in {time.time()-t0:.1f}s")

    t1 = time.time()
    outs = llm.classify(prompts)
    print(f"[score] scored {len(prompts)} in {time.time()-t1:.1f}s")

    # 写结果
    with open(a.out, "w") as f:
        for rec, ro in zip(all_records, outs):
            cand = extract_candidates(ro)
            out_rec = {
                "group": rec["_group"],
                "char": rec.get("char", ""),
                "char_id": rec.get("char_id", ""),
                "case_id": rec.get("case_id", ""),
                "lang": rec.get("lang", ""),
                "model": rec.get("model", ""),
                "sampling": rec.get("sampling", {}),
                "scores": cand,
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"[done] wrote {len(all_records)} -> {a.out}")

    # 统计
    by_group = {}
    for rec, ro in zip(all_records, outs):
        cand = extract_candidates(ro)
        score = cand.get("probs_0", cand.get("data_0", 0.0))
        by_group.setdefault(rec["_group"], []).append(score)

    print("\n=== RM Score Summary ===")
    for g in groups:
        scores = by_group.get(g, [])
        if scores:
            arr = np.array(scores)
            print(f"  {g:15s}: mean={arr.mean():.4f} std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f} n={len(arr)}")


if __name__ == "__main__":
    main()
