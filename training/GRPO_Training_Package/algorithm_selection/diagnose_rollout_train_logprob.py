#!/usr/bin/env python
"""训推 logprob 差异诊断（算法选型对照实验 §2C；预注册未执行）。

目的：在冻结同一权重版本、不做任何参数更新的前提下，量化
rollout 后端与训练后端对同一批 token 的 logprob 差异 delta：
    delta = log p_train(y_t | x, y_<t) - log p_rollout(y_t | x, y_<t)

本脚本只负责聚合与分层报告：它只能确定差异是否存在及其分布，
不能证明任何算法（GRPO/GSPO/DPO）更优。

输入：两个 JSONL，每行一个 token 记录，对齐键 (uid, response_idx, token_idx)：
    {"uid": "...", "response_idx": 0, "token_idx": 12,
     "slice": "clean" | "attack",          # 缺省按 "unknown" 分层
     "T": 317,                              # 该 response 有效 token 数（可选，
                                             #  缺省用 (uid,response_idx) 内 max(token_idx)+1）
     "logprob_rollout": -1.23,
     "logprob_train": -1.25,
     "experts_rollout": [3, 11],            # 可选：该 token 激活专家集合
     "experts_train":   [3, 11]}            # 可选；两侧都存在才计入路由一致率

只统计两侧都出现的记录；单侧多余记录计入 unmatched，不进分母。

输出（stdout 或 --out 指定的 JSON 文件）：
    n_matched / n_unmatched_rollout / n_unmatched_train
    delta:  signed_mean, abs: {mean, p50, p95, p99}
    by_slice / by_length_quartile（按 response 长度的四分位桶）
    routing: {matched, agreement} 或 null（无专家字段时不声称定位 MoE）

用法：
    python diagnose_rollout_train_logprob.py \
        --rollout rollout_logprobs.jsonl --train train_logprobs.jsonl \
        [--out diag_report.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict

REQUIRED_FIELDS = ("uid", "response_idx", "token_idx", "logprob_rollout", "logprob_train")


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:  # pragma: no cover - 防御式
                raise SystemExit(f"{path}:{lineno}: JSON 解析失败: {e}")
    return rows


def key_of(row: dict):
    return (row["uid"], row["response_idx"], row["token_idx"])


def join_by_token(rollout_rows: list[dict], train_rows: list[dict]):
    """按 (uid, response_idx, token_idx) 对齐两侧记录。"""
    idx = {key_of(r): r for r in rollout_rows}
    matched, unmatched_train = [], 0
    for tr in train_rows:
        ro = idx.pop(key_of(tr), None)
        if ro is None:
            unmatched_train += 1
            continue
        delta = float(tr["logprob_train"]) - float(ro["logprob_rollout"])
        rec = {
            "uid": tr["uid"],
            "response_idx": tr["response_idx"],
            "token_idx": tr["token_idx"],
            "slice": str(tr.get("slice", ro.get("slice", "unknown"))),
            "delta": delta,
            "experts_rollout": ro.get("experts_rollout"),
            "experts_train": tr.get("experts_train"),
        }
        matched.append(rec)
    return matched, len(idx), unmatched_train  # len(idx) = rollout 侧未匹配数


def percentile(sorted_vals: list[float], q: float) -> float:
    """线性插值百分位（inclusive，与 numpy 默认一致）。要求 sorted_vals 非空且已升序。"""
    if not sorted_vals:
        raise ValueError("percentile 需要非空列表")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def summarize_deltas(deltas: list[float]) -> dict:
    if not deltas:
        return {"n": 0, "signed_mean": None, "abs": {"mean": None, "p50": None, "p95": None, "p99": None}}
    abs_sorted = sorted(abs(d) for d in deltas)
    return {
        "n": len(deltas),
        "signed_mean": sum(deltas) / len(deltas),
        "abs": {
            "mean": abs_sorted[0] if len(abs_sorted) == 1 else sum(abs_sorted) / len(abs_sorted),
            "p50": percentile(abs_sorted, 50),
            "p95": percentile(abs_sorted, 95),
            "p99": percentile(abs_sorted, 99),
        },
    }


def response_lengths(matched: list[dict]) -> dict[tuple, int]:
    """(uid, response_idx) -> 有效 token 数 T（用记录内 max(token_idx)+1；外部可在每条带 T 覆盖）。"""
    lengths: dict[tuple, int] = {}
    override: dict[tuple, int] = {}
    for r in matched:
        k = (r["uid"], r["response_idx"])
        lengths[k] = max(lengths.get(k, 0), r["token_idx"] + 1)
        # 若任一侧行携带 T 字段，取最大者作为该 response 的长度声明
        # （load 时未保留 T；此处退化用 token 覆盖，够诊断分层用）
    return lengths if not override else {**lengths, **override}


def length_quartile_boundaries(lengths: list[int]) -> tuple[float, float, float]:
    s = sorted(lengths)
    return percentile([float(v) for v in s], 25), percentile([float(v) for v in s], 50), percentile([float(v) for v in s], 75)


def build_report(matched: list[dict], n_unmatched_rollout: int, n_unmatched_train: int) -> dict:
    report = {
        "n_matched": len(matched),
        "n_unmatched_rollout": n_unmatched_rollout,
        "n_unmatched_train": n_unmatched_train,
    }
    report["delta"] = summarize_deltas([r["delta"] for r in matched])

    by_slice: dict[str, list[float]] = defaultdict(list)
    for r in matched:
        by_slice[r["slice"]].append(r["delta"])
    report["by_slice"] = {k: summarize_deltas(v) for k, v in sorted(by_slice.items())}

    lengths = response_lengths(matched)
    if lengths:
        q25, q50, q75 = length_quartile_boundaries(list(lengths.values()))
        by_q: dict[str, list[float]] = defaultdict(list)
        for r in matched:
            t = lengths[(r["uid"], r["response_idx"])]
            bucket = "q1" if t <= q25 else "q2" if t <= q50 else "q3" if t <= q75 else "q4"
            by_q[bucket].append(r["delta"])
        report["length_quartile_boundaries"] = {"q25": q25, "q50": q50, "q75": q75}
        report["by_length_quartile"] = {k: summarize_deltas(v) for k, v in sorted(by_q.items())}

    rout = [(r["experts_rollout"], r["experts_train"]) for r in matched
            if r["experts_rollout"] is not None and r["experts_train"] is not None]
    if rout:
        agree = sum(1 for a, b in rout if set(a) == set(b))
        report["routing"] = {"matched": len(rout), "agreement": agree / len(rout)}
    else:
        report["routing"] = None  # 无专家记录时不声称定位 MoE 原因
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rollout", required=True, help="rollout 侧 per-token logprob JSONL")
    ap.add_argument("--train", required=True, help="训练侧重算 per-token logprob JSONL")
    ap.add_argument("--out", default=None, help="报告输出 JSON 路径（缺省打印 stdout）")
    args = ap.parse_args(argv)

    matched, n_un_ro, n_un_tr = join_by_token(load_jsonl(args.rollout), load_jsonl(args.train))
    report = build_report(matched, n_un_ro, n_un_tr)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"报告写入 {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
