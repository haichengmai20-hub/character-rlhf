#!/usr/bin/env python3
"""analyze_diversity.py
对比 base / step0 / step0_hot / step100 四组的多样性指标:
  1. 开场集中度: 前50字的 n-gram 重叠率
  2. 词汇丰富度: TTR (type-token ratio)
  3. 自重复: 同组内 response 的平均 pairwise 相似度
  4. 按角色/语言分组统计
"""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def ngrams(text, n=4):
    """提取字符级 n-gram"""
    chars = list(text)
    return [tuple(chars[i:i+n]) for i in range(len(chars)-n+1)]


def ngram_set(text, n=4):
    return set(ngrams(text, n))


def jaccard(s1, s2):
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def ttr(text):
    """Type-Token Ratio (字符级)"""
    chars = list(text.replace(' ', ''))
    if not chars:
        return 0.0
    return len(set(chars)) / len(chars)


def openning_ngram_overlap(records, n=4, prefix_len=50):
    """开场集中度: 所有 response 前 prefix_len 字符的 n-gram 平均 pairwise Jaccard"""
    openings = []
    for r in records:
        text = r.get("response", "")[:prefix_len]
        if text:
            openings.append(ngram_set(text, n))
    if len(openings) < 2:
        return 0.0
    sims = []
    for i in range(len(openings)):
        for j in range(i+1, len(openings)):
            sims.append(jaccard(openings[i], openings[j]))
    return np.mean(sims)


def avg_pairwise_sim(records, n=4):
    """全文本 pairwise n-gram Jaccard（采样计算，避免 O(N^2)）"""
    ngram_sets = [ngram_set(r.get("response", ""), n) for r in records]
    ngram_sets = [s for s in ngram_sets if s]
    if len(ngram_sets) < 2:
        return 0.0
    # 如果超过 50 条，随机采样 50 条
    if len(ngram_sets) > 50:
        idx = np.random.choice(len(ngram_sets), 50, replace=False)
        ngram_sets = [ngram_sets[i] for i in idx]
    sims = []
    for i in range(len(ngram_sets)):
        for j in range(i+1, len(ngram_sets)):
            sims.append(jaccard(ngram_sets[i], ngram_sets[j]))
    return np.mean(sims)


def per_char_diversity(records):
    """按角色统计多样性"""
    by_char = defaultdict(list)
    for r in records:
        by_char[r.get("char_id", "unknown")].append(r)
    results = {}
    for char, recs in by_char.items():
        results[char] = {
            "n": len(recs),
            "opening_overlap": openning_ngram_overlap(recs),
            "avg_ttr": np.mean([ttr(r["response"]) for r in recs if r.get("response")]),
        }
    return results


def main():
    data_dir = "/root/outputs/v67_step0_vs_step100"
    
    groups = {
        "base": load_jsonl(os.path.join(data_dir, "base_probe.jsonl")),
        "step0": load_jsonl(os.path.join(data_dir, "step0_probe.jsonl")),
        "step0_hot": load_jsonl(os.path.join(data_dir, "step0_probe_hot.jsonl")),
        "step100": load_jsonl(os.path.join(data_dir, "step100_probe.jsonl")),
    }

    # 加载 RM 分数
    rm_scores = {}
    rm_path = os.path.join(data_dir, "rm_scores.jsonl")
    if os.path.exists(rm_path):
        with open(rm_path) as f:
            for line in f:
                r = json.loads(line)
                rm_scores[(r["group"], r["case_id"])] = r["scores"].get("probs_0", 0.0)
    
    rm_hot_path = os.path.join(data_dir, "rm_scores_hot.jsonl")
    if os.path.exists(rm_hot_path):
        with open(rm_hot_path) as f:
            for line in f:
                r = json.loads(line)
                rm_scores[("step0_hot", r["case_id"])] = r["scores"].get("probs_0", 0.0)

    print("=" * 70)
    print("PROBE 多样性对比: base vs step0 vs step0_hot vs step100")
    print("=" * 70)

    # 全局指标
    print(f"\n{'指标':<25} {'base':>10} {'step0':>10} {'step0_hot':>10} {'step100':>10}")
    print("-" * 65)

    # 1. 开场集中度 (4-gram, 前50字)
    opening_overlaps = {}
    for g, recs in groups.items():
        opening_overlaps[g] = openning_ngram_overlap(recs, n=4, prefix_len=50)
    print(f"{'开场集中度(4g,50c)':<25} {opening_overlaps['base']:>10.4f} {opening_overlaps['step0']:>10.4f} {opening_overlaps['step0_hot']:>10.4f} {opening_overlaps['step100']:>10.4f}")
    # 越低 = 越多样

    # 2. 词汇丰富度 TTR
    ttrs = {}
    for g, recs in groups.items():
        ttrs[g] = np.mean([ttr(r["response"]) for r in recs if r.get("response")])
    print(f"{'TTR(字符级)':<25} {ttrs['base']:>10.4f} {ttrs['step0']:>10.4f} {ttrs['step0_hot']:>10.4f} {ttrs['step100']:>10.4f}")
    # 越高 = 越多样

    # 3. 全文本 pairwise 相似度
    pairwise_sims = {}
    np.random.seed(42)
    for g, recs in groups.items():
        pairwise_sims[g] = avg_pairwise_sim(recs, n=4)
    print(f"{'Pairwise Jaccard(4g)':<25} {pairwise_sims['base']:>10.4f} {pairwise_sims['step0']:>10.4f} {pairwise_sims['step0_hot']:>10.4f} {pairwise_sims['step100']:>10.4f}")
    # 越低 = 越多样

    # 4. RM 分数
    rm_means = {}
    for g in groups:
        scores = [rm_scores.get((g, r["case_id"]), None) for r in groups[g]]
        scores = [s for s in scores if s is not None]
        rm_means[g] = np.mean(scores) if scores else 0.0
    print(f"{'RM probs_0':<25} {rm_means['base']:>10.4f} {rm_means['step0']:>10.4f} {rm_means['step0_hot']:>10.4f} {rm_means['step100']:>10.4f}")

    # 5. Response 平均长度
    lens = {}
    for g, recs in groups.items():
        lens[g] = np.mean([len(r.get("response", "")) for r in recs])
    print(f"{'Avg response len':<25} {lens['base']:>10.0f} {lens['step0']:>10.0f} {lens['step0_hot']:>10.0f} {lens['step100']:>10.0f}")

    # 按角色统计开场集中度
    print(f"\n{'='*70}")
    print("按角色分组 — 开场集中度(越低越多样)")
    print(f"{'='*70}")
    
    all_chars = set()
    char_data = {}
    for g, recs in groups.items():
        char_data[g] = per_char_diversity(recs)
        all_chars.update(char_data[g].keys())
    
    print(f"{'角色':<25} {'base':>10} {'step0':>10} {'step0_hot':>10} {'step100':>10}")
    print("-" * 65)
    for char in sorted(all_chars):
        vals = []
        for g in groups:
            v = char_data[g].get(char, {}).get("opening_overlap", float('nan'))
            vals.append(v)
        print(f"{char:<25} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {vals[3]:>10.4f}")

    # 总结
    print(f"\n{'='*70}")
    print("总结")
    print(f"{'='*70}")
    
    # 开场集中度改善
    improvement = opening_overlaps["step0"] - opening_overlaps["step0_hot"]
    pct = (improvement / opening_overlaps["step0"] * 100) if opening_overlaps["step0"] > 0 else 0
    print(f"  step0 → step0_hot 开场集中度: {opening_overlaps['step0']:.4f} → {opening_overlaps['step0_hot']:.4f} (降低 {pct:.1f}%)")
    
    if opening_overlaps["step0_hot"] < opening_overlaps["step0"]:
        print(f"  ✓ 高温采样显著降低开场集中度，多样性回来了！")
    else:
        print(f"  ✗ 高温采样未能降低开场集中度")
    
    base_vs_hot = opening_overlaps["base"] - opening_overlaps["step0_hot"]
    if abs(base_vs_hot) < 0.01:
        print(f"  ≈ step0_hot 开场集中度已接近 base 水平")
    elif base_vs_hot > 0:
        print(f"  base 仍比 step0_hot 更多样（差 {base_vs_hot:.4f}）")
    else:
        print(f"  step0_hot 比 base 更多样（差 {-base_vs_hot:.4f}）")


if __name__ == "__main__":
    main()
