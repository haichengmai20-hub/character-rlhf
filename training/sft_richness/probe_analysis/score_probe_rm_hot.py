#!/usr/bin/env python3
"""score_probe_rm_hot.py
用 RM 给 step0_probe_hot.jsonl 打分，输出格式与 rm_scores.jsonl 一致便于对比。
"""
import argparse, json, os, time
import numpy as np


def load_probe(path):
    records = []
    with open(path) as f:
        for line in f:
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
    ap.add_argument("--probe", default="/root/outputs/v67_step0_vs_step100/step0_probe_hot.jsonl")
    ap.add_argument("--rm", default="/root/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_v6.6_fg_7gpu_merged_vllm")
    ap.add_argument("--out", default="/root/outputs/v67_step0_vs_step100/rm_scores_hot.jsonl")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()

    from vllm import LLM
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.rm, trust_remote_code=True)
    records = load_probe(a.probe)
    print(f"[load] {len(records)} records from {a.probe}")

    prompts = []
    for rec in records:
        msgs = build_messages(rec)
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompts.append(rendered)

    print(f"[score] total={len(prompts)} tp={a.tp}")

    t0 = time.time()
    llm = LLM(
        model=a.rm, runner="pooling", tensor_parallel_size=a.tp,
        gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_len,
        limit_mm_per_prompt={"image": 0, "video": 0},
        trust_remote_code=True, enforce_eager=True,
    )
    print(f"[load] RM loaded in {time.time()-t0:.1f}s")

    t1 = time.time()
    outs = llm.classify(prompts)
    print(f"[score] scored {len(prompts)} in {time.time()-t1:.1f}s")

    print("[probe] first output candidates:", extract_candidates(outs[0]))

    with open(a.out, "w") as f:
        for rec, ro in zip(records, outs):
            cand = extract_candidates(ro)
            out_rec = {
                "group": rec.get("model", "step0_hot"),
                "char": rec.get("char", ""),
                "char_id": rec.get("char_id", ""),
                "case_id": rec.get("case_id", ""),
                "lang": rec.get("lang", ""),
                "model": rec.get("model", ""),
                "scores": cand,
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"[done] wrote {len(records)} -> {a.out}")

    scores = [extract_candidates(ro).get("probs_0", 0.0) for ro in outs]
    arr = np.array(scores)
    print(f"\n=== step0_hot RM Score ===")
    print(f"  mean={arr.mean():.4f} std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f} n={len(arr)}")


if __name__ == "__main__":
    main()
