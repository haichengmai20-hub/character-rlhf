#!/usr/bin/env python3
"""
训练指标可视化脚本 — 解析 OpenRLHF RM 和 verl GRPO 训练日志，生成仪表板图。

支持的日志格式:
  - OpenRLHF RM:   save_path/train.log (tee 输出)
  - verl GRPO:     save_dir/*/trainer.log 或控制台重定向文件

输出图表:
  1. loss_curve.png        — RM 训练 loss + reward_accuracy
  2. grpo_metrics.png      — Actor reward / KL / entropy / response_length
  3. moe_load_history.png  — 来自 MoELoadCallback 的每步负载趋势（若存在）

用法:
  # RM 训练后
  python plot_training_metrics.py \
    --rm_log saves/qwen3-30b-moe/chatjoy_rm_v1/train.log \
    --output_dir saves/qwen3-30b-moe/chatjoy_rm_v1/plots/

  # verl GRPO 训练后
  python plot_training_metrics.py \
    --verl_log saves/verl/chatjoy_grpo_lora/grpo_train.log \
    --moe_log  saves/verl/chatjoy_grpo_lora/moe_logs/moe_load_history.jsonl \
    --output_dir saves/verl/chatjoy_grpo_lora/plots/

  # 同时绘制两者
  python plot_training_metrics.py \
    --rm_log   saves/.../train.log \
    --verl_log saves/.../grpo_train.log \
    --output_dir plots/
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1. OpenRLHF RM 日志解析
# ─────────────────────────────────────────────────────────────────────────────
# 典型行格式（来自 openrlhf.trainer.rm_trainer）:
#   step 5/270, loss: 0.6823, acc: 0.5208, lr: 8.91e-06
#   step 50/270, loss: 0.4512, acc: 0.6875, lr: 8.43e-06
# eval 行:
#   Eval: step 50, acc/mean: 0.6875, loss/mean: 0.4512

_RM_TRAIN_RE = re.compile(
    r"step\s+(\d+).*?loss[:\s]+([\d.eE+\-]+).*?acc[:\s]+([\d.eE+\-]+).*?lr[:\s]+([\d.eE+\-]+)",
    re.IGNORECASE,
)
_RM_EVAL_RE = re.compile(
    r"[Ee]val.*?step\s+(\d+).*?acc.*?:\s*([\d.eE+\-]+).*?loss.*?:\s*([\d.eE+\-]+)",
    re.IGNORECASE,
)
# Also catch simpler patterns like "{'loss': ..., 'acc': ..., 'lr': ..., 'step': ...}"
_RM_DICT_RE = re.compile(
    r"'step':\s*(\d+).*?'loss':\s*([\d.eE+\-]+).*?'acc':\s*([\d.eE+\-]+).*?'lr':\s*([\d.eE+\-]+)",
)


def parse_rm_log(log_path: str) -> Tuple[List[dict], List[dict]]:
    """
    返回 (train_records, eval_records)
    每个 record: {"step": int, "loss": float, "acc": float, "lr": float}
    """
    train_records, eval_records = [], []

    with open(log_path) as f:
        for line in f:
            # eval
            m = _RM_EVAL_RE.search(line)
            if m:
                eval_records.append({
                    "step": int(m.group(1)),
                    "acc":  float(m.group(2)),
                    "loss": float(m.group(3)),
                })
                continue
            # train (格式1)
            m = _RM_TRAIN_RE.search(line)
            if m:
                train_records.append({
                    "step": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "acc":  float(m.group(3)),
                    "lr":   float(m.group(4)),
                })
                continue
            # train (格式2: dict-like)
            m = _RM_DICT_RE.search(line)
            if m:
                train_records.append({
                    "step": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "acc":  float(m.group(3)),
                    "lr":   float(m.group(4)),
                })

    return train_records, eval_records


# ─────────────────────────────────────────────────────────────────────────────
# 2. verl GRPO 日志解析
# ─────────────────────────────────────────────────────────────────────────────
# verl 通过 wandb / console logger 输出，典型 key 名:
#   train/reward_mean, train/kl, train/entropy, train/response_length
#   train/actor_loss, train/critic_loss, train/pg_loss, train/vf_loss
# console logger 每步输出一行 JSON 或 key=val 格式

_VERL_JSON_RE = re.compile(r"^\{.*\}$")
_VERL_KV_RE   = re.compile(r"([\w/\.]+)=([\d.eE+\-]+)")
_VERL_STEP_RE = re.compile(r"\bstep[:\s=]+(\d+)\b", re.IGNORECASE)


def parse_verl_log(log_path: str) -> List[dict]:
    """
    返回每步的指标 dict，尽力解析控制台日志。
    """
    records = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = {}

            # 尝试 JSON 行
            if line.startswith("{"):
                try:
                    d = json.loads(line)
                    record.update({k: v for k, v in d.items() if isinstance(v, (int, float))})
                except Exception:
                    pass

            # 尝试 key=val 格式
            for m in _VERL_KV_RE.finditer(line):
                try:
                    record[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass

            if not record:
                continue

            # 标准化 step
            step_m = _VERL_STEP_RE.search(line)
            if step_m:
                record["step"] = int(step_m.group(1))
            elif "step" not in record and "global_step" in record:
                record["step"] = int(record["global_step"])

            if record:
                records.append(record)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 3. RM eval reward 实例解析（来自 rm_trainer.py 写出的 eval_rewards/step_*.jsonl）
# ─────────────────────────────────────────────────────────────────────────────

def parse_eval_reward_dir(eval_rewards_dir: str) -> List[dict]:
    """
    扫描 eval_rewards/ 目录下所有 step_*.jsonl，
    返回每步的聚合统计 dict: {step, acc, gap_mean, gap_p10, gap_p50, gap_p90,
                               chosen_mean, rejected_mean, neg_gap_frac, sat_frac}
    """
    import glob
    step_stats = []
    pattern = os.path.join(eval_rewards_dir, "step_*.jsonl")
    for fpath in sorted(glob.glob(pattern)):
        step_str = os.path.basename(fpath).replace("step_", "").replace(".jsonl", "")
        try:
            step = int(step_str)
        except ValueError:
            continue
        chosen_list, rejected_list, gap_list = [], [], []
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    chosen_list.append(r["chosen"])
                    rejected_list.append(r["rejected"])
                    gap_list.append(r["gap"])
                except Exception:
                    pass
        if not gap_list:
            continue
        gap_arr = sorted(gap_list)
        n = len(gap_arr)
        step_stats.append({
            "step":          step,
            "n":             n,
            "acc":           sum(1 for g in gap_list if g > 0) / n,
            "gap_mean":      sum(gap_list) / n,
            "gap_p10":       gap_arr[max(0, int(n * 0.10))],
            "gap_p50":       gap_arr[int(n * 0.50)],
            "gap_p90":       gap_arr[min(n-1, int(n * 0.90))],
            "chosen_mean":   sum(chosen_list) / n,
            "rejected_mean": sum(rejected_list) / n,
            "neg_gap_frac":  sum(1 for g in gap_list if g < 0) / n,
            "sat_frac":      sum(1 for c in chosen_list if abs(c) > 5) / n,
        })
    return step_stats


def plot_eval_rewards(step_stats: List[dict], output_dir: Path):
    if not step_stats:
        print("No eval reward data found, skipping")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps       = [s["step"]          for s in step_stats]
    acc         = [s["acc"]           for s in step_stats]
    gap_mean    = [s["gap_mean"]      for s in step_stats]
    gap_p10     = [s["gap_p10"]       for s in step_stats]
    gap_p90     = [s["gap_p90"]       for s in step_stats]
    chosen_mean = [s["chosen_mean"]   for s in step_stats]
    rej_mean    = [s["rejected_mean"] for s in step_stats]
    neg_frac    = [s["neg_gap_frac"]  for s in step_stats]
    sat_frac    = [s["sat_frac"]      for s in step_stats]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RM Eval Reward Instances", fontsize=14)

    # 准确率
    ax = axes[0, 0]
    ax.plot(steps, acc, "g-o", markersize=5, label="acc")
    ax.axhline(0.70, color="g", linestyle="--", alpha=0.5, label="target 0.70")
    ax.axhline(0.60, color="orange", linestyle="--", alpha=0.5, label="warn 0.60")
    ax.fill_between(steps, [0.6]*len(steps), acc,
                    where=[a > 0.6 for a in acc], alpha=0.1, color="green")
    ax.set_ylim(0.4, 1.0); ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
    ax.set_title("Reward Accuracy (chosen > rejected)"); ax.legend(); ax.grid(True, alpha=0.3)

    # gap 分布趋势
    ax = axes[0, 1]
    ax.plot(steps, gap_mean, "b-o", markersize=5, label="gap mean")
    ax.fill_between(steps, gap_p10, gap_p90, alpha=0.2, color="blue", label="p10–p90")
    ax.axhline(0, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("Step"); ax.set_ylabel("Reward Gap (chosen − rejected)")
    ax.set_title("Gap Distribution over Steps"); ax.legend(); ax.grid(True, alpha=0.3)

    # 绝对分数
    ax = axes[1, 0]
    ax.plot(steps, chosen_mean, "g-o", markersize=5, label="chosen mean")
    ax.plot(steps, rej_mean,    "r-o", markersize=5, label="rejected mean")
    ax.axhline( 5, color="gray", linestyle=":", alpha=0.5, label="saturation ±5")
    ax.axhline(-5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Step"); ax.set_ylabel("Score")
    ax.set_title("Absolute Reward Scores"); ax.legend(); ax.grid(True, alpha=0.3)

    # 异常比例
    ax = axes[1, 1]
    ax.plot(steps, [f*100 for f in neg_frac],  "r-o", markersize=5, label="wrong pairs % (gap<0)")
    ax.plot(steps, [f*100 for f in sat_frac],  "m--o", markersize=5, label="saturated % (|score|>5)")
    ax.axhline(10, color="orange", linestyle="--", alpha=0.5, label="warn 10%")
    ax.set_xlabel("Step"); ax.set_ylabel("%")
    ax.set_title("Anomaly Rates"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "eval_reward_instances.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # 打印最新一步的诊断
    last = step_stats[-1]
    print(f"\n  [Latest eval step={last['step']}]")
    print(f"  acc={last['acc']:.3f}  gap_mean={last['gap_mean']:+.3f}  "
          f"gap_p10={last['gap_p10']:+.3f}  gap_p90={last['gap_p90']:+.3f}")
    print(f"  chosen_mean={last['chosen_mean']:+.3f}  rejected_mean={last['rejected_mean']:+.3f}")
    print(f"  wrong pairs={last['neg_gap_frac']*100:.1f}%  "
          f"saturated={last['sat_frac']*100:.1f}%")
    if last["neg_gap_frac"] > 0.10:
        print("  !!! > 10% pairs scored wrong — RM quality issue !!!")
    if last["sat_frac"] > 0.10:
        print("  !!! > 10% scores saturated (|score| > 5) — gradient vanishing risk !!!")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MoE 负载历史解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_moe_log(log_path: str) -> List[dict]:
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 4. 绘图
# ─────────────────────────────────────────────────────────────────────────────

def plot_rm_training(train_records: List[dict], eval_records: List[dict], output_dir: Path):
    if not train_records:
        print("No RM train records found, skipping RM plot")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_steps = [r["step"] for r in train_records]
    train_loss  = [r["loss"] for r in train_records]
    train_acc   = [r["acc"]  for r in train_records]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("OpenRLHF RM Training", fontsize=14)

    # Loss
    ax = axes[0, 0]
    ax.plot(train_steps, train_loss, "b-", linewidth=0.8, alpha=0.7, label="Train Loss")
    if eval_records:
        ax.scatter([r["step"] for r in eval_records],
                   [r["loss"] for r in eval_records],
                   color="red", zorder=5, s=30, label="Eval Loss")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(train_steps, train_acc, "g-", linewidth=0.8, alpha=0.7, label="Train Acc")
    if eval_records:
        ax.scatter([r["step"] for r in eval_records],
                   [r["acc"] for r in eval_records],
                   color="red", zorder=5, s=30, label="Eval Acc")
    ax.axhline(0.7, color="g", linestyle="--", alpha=0.5, label="Target 0.70")
    ax.axhline(0.6, color="orange", linestyle="--", alpha=0.5, label="Min 0.60")
    ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
    ax.set_title("Reward Accuracy"); ax.legend(); ax.grid(True, alpha=0.3)

    # LR
    ax = axes[1, 0]
    if train_records and "lr" in train_records[0]:
        lrs = [r["lr"] for r in train_records]
        ax.plot(train_steps, lrs, "m-", linewidth=0.8)
        ax.set_xlabel("Step"); ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule"); ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "LR not available", ha="center", va="center", transform=ax.transAxes)

    # Smoothed loss
    ax = axes[1, 1]
    if len(train_loss) >= 10:
        window = max(5, len(train_loss) // 50)
        smoothed = np.convolve(train_loss, np.ones(window)/window, mode="valid")
        smooth_steps = train_steps[window-1:]
        ax.plot(smooth_steps, smoothed, "b-", linewidth=1.5, label=f"Loss (smooth w={window})")
    else:
        ax.plot(train_steps, train_loss, "b-")
    if eval_records:
        ax.scatter([r["step"] for r in eval_records],
                   [r["loss"] for r in eval_records],
                   color="red", zorder=5, s=30, label="Eval Loss")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss (smoothed)")
    ax.set_title("Smoothed Loss Curve"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "loss_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # 打印最终统计
    if train_records:
        last_n = min(20, len(train_records))
        avg_loss = np.mean([r["loss"] for r in train_records[-last_n:]])
        avg_acc  = np.mean([r["acc"]  for r in train_records[-last_n:]])
        print(f"  RM Training — Last {last_n} steps: loss={avg_loss:.4f}, acc={avg_acc:.4f}")
    if eval_records:
        last_eval = eval_records[-1]
        print(f"  RM Eval — Last checkpoint: step={last_eval['step']}, acc={last_eval['acc']:.4f}, loss={last_eval['loss']:.4f}")


def plot_grpo_training(records: List[dict], output_dir: Path):
    if not records:
        print("No verl GRPO records found, skipping GRPO plot")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 归一化 key 名（verl 不同版本 key 有差异）
    KEY_ALIASES = {
        "reward":           ["train/reward_mean", "reward_mean", "critic/reward_mean", "train/reward"],
        "kl":               ["train/kl", "kl_loss", "actor/kl", "train/kl_loss"],
        "entropy":          ["train/entropy", "actor/entropy", "entropy"],
        "response_length":  ["train/response_length", "response_length", "train/gen_len"],
        "actor_loss":       ["train/actor_loss", "actor_loss", "train/pg_loss", "pg_loss"],
        "critic_loss":      ["train/critic_loss", "critic_loss", "train/vf_loss"],
    }

    def extract(records, aliases):
        for key in aliases:
            vals = [r.get(key) for r in records if r.get(key) is not None]
            steps = [r.get("step", i) for i, r in enumerate(records) if r.get(key) is not None]
            if vals:
                return steps, vals
        return [], []

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle("verl GRPO Actor Training", fontsize=14)

    plots = [
        ("Reward (mean)",     KEY_ALIASES["reward"],          axes[0, 0], "g"),
        ("KL Divergence",     KEY_ALIASES["kl"],               axes[0, 1], "r"),
        ("Entropy",           KEY_ALIASES["entropy"],          axes[1, 0], "b"),
        ("Response Length",   KEY_ALIASES["response_length"],  axes[1, 1], "m"),
        ("Actor/PG Loss",     KEY_ALIASES["actor_loss"],       axes[2, 0], "darkorange"),
        ("Critic/VF Loss",    KEY_ALIASES["critic_loss"],      axes[2, 1], "teal"),
    ]

    for title, aliases, ax, color in plots:
        steps, vals = extract(records, aliases)
        if vals:
            ax.plot(steps, vals, color=color, linewidth=0.8, alpha=0.8)
            # KL 预警线
            if "KL" in title:
                ax.axhline(0.1, color="orange", linestyle="--", alpha=0.6, label="Warn 0.1")
                ax.axhline(0.3, color="red",    linestyle="--", alpha=0.6, label="Alert 0.3")
                ax.legend(fontsize=8)
            ax.set_xlabel("Step"); ax.set_ylabel(title)
            ax.set_title(title); ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"{title}\nnot found in log",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_title(title)

    plt.tight_layout()
    out_path = output_dir / "grpo_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_moe_history(records: List[dict], output_dir: Path):
    if not records:
        print("No MoE load history records found, skipping MoE plot")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r.get("step", i) for i, r in enumerate(records)]

    def safe_extract(key):
        return [r.get(key) for r in records if r.get(key) is not None]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MoE Expert Load Balance over Training", fontsize=14)

    ax = axes[0, 0]
    vals = safe_extract("entropy_norm_mean")
    if vals:
        ax.plot(steps[:len(vals)], vals, "b-o", markersize=4)
        ax.axhline(0.90, color="g", linestyle="--", label="Target ≥ 0.90")
        ax.axhline(0.80, color="r", linestyle="--", label="Alert < 0.80")
        ax.set_ylim(0, 1.05)
    ax.set_xlabel("Step"); ax.set_ylabel("Entropy (mean)")
    ax.set_title("Normalized Entropy (mean over layers)"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    vals = safe_extract("entropy_norm_min")
    if vals:
        ax.plot(steps[:len(vals)], vals, "b--o", markersize=4, alpha=0.7)
        ax.axhline(0.80, color="r", linestyle="--", label="Alert < 0.80")
        ax.set_ylim(0, 1.05)
    ax.set_xlabel("Step"); ax.set_ylabel("Entropy (min)")
    ax.set_title("Normalized Entropy (min layer = worst)"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    vals = safe_extract("cv_mean")
    if vals:
        ax.plot(steps[:len(vals)], vals, "r-o", markersize=4)
        ax.axhline(0.20, color="g", linestyle="--", label="Target ≤ 0.20")
        ax.axhline(0.50, color="r", linestyle="--", label="Alert > 0.50")
    ax.set_xlabel("Step"); ax.set_ylabel("CV (mean)")
    ax.set_title("Load CV (std/mean, mean over layers)"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    vals = safe_extract("imbalance_max")
    if vals:
        ax.plot(steps[:len(vals)], vals, "darkorange", marker="o", markersize=4)
        ax.axhline(5.0, color="r", linestyle="--", label="Alert > 5.0")
    ax.set_xlabel("Step"); ax.set_ylabel("max_load/mean_load (max layer)")
    ax.set_title("Peak Imbalance Ratio"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "moe_load_history.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # 打印趋势
    if len(records) >= 2:
        first, last = records[0], records[-1]
        print(f"  MoE entropy_norm: {first.get('entropy_norm_mean','?')} → {last.get('entropy_norm_mean','?')}")
        print(f"  MoE cv_mean:      {first.get('cv_mean','?')} → {last.get('cv_mean','?')}")
        imb_max_last = last.get("imbalance_max", "?")
        if isinstance(imb_max_last, float) and imb_max_last > 5.0:
            print(f"  !!! 负载不均衡预警: imbalance_max={imb_max_last:.2f} > 5.0 !!!")


# ─────────────────────────────────────────────────────────────────────────────
# 5. 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="训练指标可视化")
    parser.add_argument("--rm_log",           default="", help="OpenRLHF RM train.log 路径")
    parser.add_argument("--eval_rewards_dir", default="", help="RM eval_rewards/ 目录（含 step_*.jsonl）")
    parser.add_argument("--verl_log",         default="", help="verl GRPO 训练日志路径")
    parser.add_argument("--moe_log",          default="", help="MoELoadCallback 输出的 JSONL 路径")
    parser.add_argument("--output_dir",       default="plots", help="图表输出目录")
    # 便捷模式：直接传 save_path，自动推断所有路径
    parser.add_argument("--save_path",        default="", help="OpenRLHF RM save_path（自动推断 train.log + eval_rewards/）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 从 save_path 自动推断路径
    if args.save_path:
        sp = Path(args.save_path)
        if not args.rm_log and (sp / "train.log").exists():
            args.rm_log = str(sp / "train.log")
        if not args.eval_rewards_dir and (sp / "eval_rewards").is_dir():
            args.eval_rewards_dir = str(sp / "eval_rewards")
        if not args.output_dir or args.output_dir == "plots":
            output_dir = sp / "plots"
            output_dir.mkdir(parents=True, exist_ok=True)

    if args.rm_log and Path(args.rm_log).exists():
        print(f"\n[RM] Parsing {args.rm_log} ...")
        train_r, eval_r = parse_rm_log(args.rm_log)
        print(f"  Found {len(train_r)} train records, {len(eval_r)} eval records")
        plot_rm_training(train_r, eval_r, output_dir)
    elif args.rm_log:
        print(f"[RM] Log file not found: {args.rm_log}")

    if args.eval_rewards_dir and Path(args.eval_rewards_dir).is_dir():
        print(f"\n[Eval Rewards] Scanning {args.eval_rewards_dir} ...")
        step_stats = parse_eval_reward_dir(args.eval_rewards_dir)
        print(f"  Found {len(step_stats)} eval checkpoints")
        plot_eval_rewards(step_stats, output_dir)
    elif args.eval_rewards_dir:
        print(f"[Eval Rewards] Dir not found: {args.eval_rewards_dir}")

    if args.verl_log and Path(args.verl_log).exists():
        print(f"\n[GRPO] Parsing {args.verl_log} ...")
        verl_records = parse_verl_log(args.verl_log)
        print(f"  Found {len(verl_records)} records")
        plot_grpo_training(verl_records, output_dir)
    elif args.verl_log:
        print(f"[GRPO] Log file not found: {args.verl_log}")

    if args.moe_log and Path(args.moe_log).exists():
        print(f"\n[MoE] Parsing {args.moe_log} ...")
        moe_records = parse_moe_log(args.moe_log)
        print(f"  Found {len(moe_records)} MoE records")
        plot_moe_history(moe_records, output_dir)
    elif args.moe_log:
        print(f"[MoE] Log file not found: {args.moe_log}")

    if not any([args.rm_log, args.verl_log, args.moe_log]):
        parser.print_help()

    # 自动扫描常用路径
    if not any([args.rm_log, args.verl_log, args.moe_log]):
        return

    print(f"\nAll plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
