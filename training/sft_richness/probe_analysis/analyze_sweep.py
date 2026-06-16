#!/usr/bin/env python3
"""analyze_sweep.py
综合分析 sweep 结果: 
  已有组: base(t0.7), step0(t0.7), step0_hot(t1.0), step100(t0.7)
  新增组: step0_t0.8, step0_t0.85, step0_t0.9, step100_t1.0

输出: 开场集中度 / TTR / Pairwise Jaccard / RM probs_0
"""
import json, os, sys
from collections import Counter
from itertools import combinations
import numpy as np

DATA_DIR = "/root/outputs/v67_step0_vs_step100"

# 定义所有组及其 jsonl 文件
GROUPS = [
    ("base",          "base_probe.jsonl"),
    ("step0",         "step0_probe.jsonl"),
    ("step0_t0.8",    "step0_t0.8.jsonl"),
    ("step0_t0.85",   "step0_t0.85.jsonl"),
    ("step0_t0.9",    "step0_t0.9.jsonl"),
    ("step0_hot",     "step0_probe_hot.jsonl"),
    ("step100",       "step100_probe.jsonl"),
    ("step100_t1.0",  "step100_t1.0.jsonl"),
]

RM_FILES = {
    "base":      "rm_scores.jsonl",
    "step0":     "rm_scores.jsonl",
    "step0_hot": "rm_scores_hot.jsonl",
    "step100":   "rm_scores.jsonl",
    "step0_t0.8":    "rm_scores_sweep.jsonl",
    "step0_t0.85":   "rm_scores_sweep.jsonl",
    "step0_t0.9":    "rm_scores_sweep.jsonl",
    "step100_t1.0":  "rm_scores_sweep.jsonl",
}


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def get_ngrams(text, n=4):
    chars = list(text)
    return [tuple(chars[i:i+n]) for i in range(len(chars)-n+1)]


def opening_concentration(records, n=4, top_k=50):
    """开场集中度: top_k 最常见 n-gram 在所有 n-gram 中的占比"""
    all_ngrams = []
    for r in records:
        resp = r.get("response", "")
        opening = resp[:200] if len(resp) >= 200 else resp
        all_ngrams.extend(get_ngrams(opening, n))
    if not all_ngrams:
        return 0.0
    counts = Counter(all_ngrams)
    top_counts = sum(c for _, c in counts.most_common(top_k))
    return top_counts / len(all_ngrams)


def char_ttr(records):
    """字符级 Type-Token Ratio"""
    all_chars = []
    for r in records:
        all_chars.extend(list(r.get("response", "")))
    if not all_chars:
        return 0.0
    return len(set(all_chars)) / len(all_chars)


def pairwise_jaccard_4gram(records, sample_n=None):
    """Pairwise Jaccard similarity (4-gram)"""
    sets = []
    for r in records:
        resp = r.get("response", "")
        sets.append(set(get_ngrams(resp, 4)))
    if len(sets) < 2:
        return 0.0
    if sample_n and len(sets) > sample_n:
        import random
        idx = random.Random(42).sample(range(len(sets)), sample_n)
        sets = [sets[i] for i in idx]
    sims = []
    for i, j in combinations(range(len(sets)), 2):
        a, b = sets[i], sets[j]
        if not a and not b:
            continue
        union = len(a | b)
        if union == 0:
            continue
        sims.append(len(a & b) / union)
    return np.mean(sims) if sims else 0.0


def action_ratio(records):
    """动作描写占比 (*...* / 整体字符)"""
    import re
    total_chars = 0
    action_chars = 0
    for r in records:
        resp = r.get("response", "")
        total_chars += len(resp)
        # 匹配 *...* 动作描写
        actions = re.findall(r'\*[^*]+\*', resp)
        action_chars += sum(len(a) for a in actions)
    return action_chars / total_chars if total_chars > 0 else 0.0


RM_GROUP_MAP = {
    "base":      "base",
    "step0":     "step0",
    "step0_hot": "step0_s1_hot",
    "step100":   "step100",
    "step0_t0.8":    "step0_t0.8",
    "step0_t0.85":   "step0_t0.85",
    "step0_t0.9":    "step0_t0.9",
    "step100_t1.0":  "step100_t1.0",
}


def load_rm_scores(group_name):
    """从 RM 打分文件中提取指定组的 probs_0"""
    rm_file = RM_FILES.get(group_name)
    if not rm_file:
        return []
    path = os.path.join(DATA_DIR, rm_file)
    if not os.path.exists(path):
        return []
    rm_group = RM_GROUP_MAP.get(group_name, group_name)
    scores = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("group") == rm_group:
                s = rec.get("scores", {})
                score = s.get("probs_0", s.get("data_0", 0.0))
                scores.append(score)
    return scores


def main():
    results = {}
    for name, fname in GROUPS:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"[skip] {name}: {fname} not found")
            continue
        recs = load_jsonl(path)
        
        oc = opening_concentration(recs, n=4, top_k=50)
        ttr = char_ttr(recs)
        jacc = pairwise_jaccard_4gram(recs)
        ar = action_ratio(recs)
        
        rm_scores = load_rm_scores(name)
        rm_mean = np.mean(rm_scores) if rm_scores else float('nan')
        
        results[name] = {
            "n": len(recs),
            "opening_conc": oc,
            "ttr": ttr,
            "jaccard_4g": jacc,
            "action_ratio": ar,
            "rm_probs0": rm_mean,
        }
        print(f"[done] {name}: conc={oc:.4f} ttr={ttr:.4f} jacc={jacc:.4f} action={ar:.4f} rm={rm_mean:.4f}")

    # 打印汇总表
    print("\n" + "=" * 100)
    print(f"{'Group':15s} | {'N':>4s} | {'OpenConc(4g,50)':>15s} | {'TTR(char)':>10s} | {'Jaccard(4g)':>12s} | {'ActionR':>8s} | {'RM_p0':>8s}")
    print("-" * 100)
    
    # 按 step0 温度排序展示
    order = ["base", "step0", "step0_t0.8", "step0_t0.85", "step0_t0.9", "step0_hot", "step100", "step100_t1.0"]
    for name in order:
        if name not in results:
            continue
        r = results[name]
        print(f"{name:15s} | {r['n']:4d} | {r['opening_conc']:15.4f} | {r['ttr']:10.4f} | {r['jaccard_4g']:12.4f} | {r['action_ratio']:8.4f} | {r['rm_probs0']:8.4f}")

    # 拐点分析
    print("\n=== 拐点分析 (step0 系列) ===")
    step0_series = ["step0", "step0_t0.8", "step0_t0.85", "step0_t0.9", "step0_hot"]
    print(f"{'Temp':>6s} | {'OpenConc':>10s} | {'Delta%':>8s} | {'RM_p0':>8s} | {'Delta%':>8s} | {'ActionR':>8s}")
    prev_conc = None
    prev_rm = None
    for name in step0_series:
        if name not in results:
            continue
        r = results[name]
        # 从名字提取温度
        if name == "step0":
            temp = "0.70"
        elif name == "step0_hot":
            temp = "1.00"
        else:
            temp = name.split("t")[1]
        
        conc_delta = f"{(r['opening_conc'] - prev_conc)/prev_conc*100:+.1f}%" if prev_conc else "   -"
        rm_delta = f"{(r['rm_probs0'] - prev_rm)/prev_rm*100:+.1f}%" if prev_rm else "   -"
        print(f"{temp:>6s} | {r['opening_conc']:10.4f} | {conc_delta:>8s} | {r['rm_probs0']:8.4f} | {rm_delta:>8s} | {r['action_ratio']:8.4f}")
        prev_conc = r['opening_conc']
        prev_rm = r['rm_probs0']

    # step100 防御检验
    print("\n=== step100 防御检验 ===")
    if "step100" in results and "step100_t1.0" in results:
        s100 = results["step100"]
        s100h = results["step100_t1.0"]
        print(f"  step100  (t=0.7):  RM={s100['rm_probs0']:.4f}  OpenConc={s100['opening_conc']:.4f}  ActionR={s100['action_ratio']:.4f}")
        print(f"  step100  (t=1.0):  RM={s100h['rm_probs0']:.4f}  OpenConc={s100h['opening_conc']:.4f}  ActionR={s100h['action_ratio']:.4f}")
        print(f"  RM 变化: {(s100h['rm_probs0']-s100['rm_probs0'])/s100['rm_probs0']*100:+.1f}%")
        print(f"  集中度变化: {(s100h['opening_conc']-s100['opening_conc'])/s100['opening_conc']*100:+.1f}%")
        if s100h['rm_probs0'] >= 0.45:
            print(f"  >> 防御守住! step100 + t=1.0 是产品候选")
        else:
            print(f"  >> 防御未守住, RM 从 {s100['rm_probs0']:.4f} 降到 {s100h['rm_probs0']:.4f}")

    # 保存结果
    out_path = os.path.join(DATA_DIR, "sweep_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[done] saved -> {out_path}")


if __name__ == "__main__":
    main()
