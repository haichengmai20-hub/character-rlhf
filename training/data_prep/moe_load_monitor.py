#!/usr/bin/env python3
"""
MoE 专家负载监控工具 — Qwen3-30B-MoE (128 experts, top-8, 48 layers)

用途:
  1. 独立分析模式：加载 checkpoint，跑推理数据集，输出每层负载统计
  2. 训练回调模式：嵌入 OpenRLHF / verl 训练循环，周期性记录负载指标

核心指标:
  - load_entropy_norm  : 归一化熵 H/H_max ∈ [0,1]，越接近 1 越均衡
  - load_cv            : 变异系数 std/mean，越接近 0 越均衡
  - top1_load_frac     : 最热专家激活频率占比（应 ≈ 1/128 = 0.78%）
  - dead_expert_count  : 激活频率 < 1% × 均值的专家数
  - imbalance_ratio    : max_load / mean_load，越接近 1 越均衡

用法:
  # 独立分析
  python moe_load_monitor.py \
    --model_path saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged \
    --data_file data/openrlhf_rm/rm_eval.jsonl \
    --output_dir moe_analysis/ \
    --max_samples 200

  # 训练集成（在训练脚本中 import）
  from moe_load_monitor import MoELoadCallback
  callback = MoELoadCallback(log_every_steps=50, output_dir="saves/rm_v1/moe_logs")
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 1. Hook 机制：拦截 MoE router 输出
# ─────────────────────────────────────────────────────────────────────────────

class MoELoadCollector:
    """
    注册到模型所有 MoE 层的 forward hook，收集每步的专家激活索引。

    支持的路由层名称模式:
      - Qwen3MoE:  layer.mlp  →  gate (Linear) → topk indices
      - 通用检测:  寻找 output_shape=(batch, seq, top_k) 的整数张量
    """

    def __init__(self, model: nn.Module, num_experts: int = 128, top_k: int = 8):
        self.num_experts = num_experts
        self.top_k = top_k
        self.hooks: List = []
        # layer_idx -> list of expert_indices tensors (each shape [n_tokens, top_k])
        self._raw: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self._layer_count = 0
        self._register_hooks(model)

    def _register_hooks(self, model: nn.Module):
        """
        在所有 Qwen3MoeSparseMoeBlock（或含 .gate Linear 的 MoE 模块）上挂 hook。
        """
        layer_idx = 0
        for name, module in model.named_modules():
            # Qwen3MoE: 路由器是 MoeBlock 里的 .gate 属性（一个 Linear）
            # 通过检查父模块名称中含 "mlp" 且模块本身有 "gate" 和 "experts" 子模块来识别
            cls_name = type(module).__name__
            if cls_name in ("Qwen3MoeSparseMoeBlock", "MoEBlock", "SparseMoeBlock"):
                idx = layer_idx
                layer_idx += 1

                def make_hook(i):
                    def hook(mod, inp, out):
                        # out 可能是 (hidden, router_logits) 或 hidden
                        # 直接拦截 gate (router Linear) 更可靠，见下方
                        pass
                    return hook

                # 优先 hook gate（路由器 Linear 本身），更直接
                if hasattr(module, "gate") and isinstance(module.gate, nn.Linear):
                    gate_idx = layer_idx - 1

                    def make_gate_hook(i):
                        def gate_hook(mod, inp, out):
                            # out: [n_tokens, num_experts] logits
                            with torch.no_grad():
                                topk_idx = out.topk(self.top_k, dim=-1).indices  # [n_tokens, top_k]
                                self._raw[i].append(topk_idx.detach().cpu())
                        return gate_hook

                    h = module.gate.register_forward_hook(make_gate_hook(gate_idx))
                    self.hooks.append(h)

        self._layer_count = layer_idx
        if self._layer_count == 0:
            # fallback: 通过名称模式查找 router
            for name, module in model.named_modules():
                if name.endswith(".gate") and isinstance(module, nn.Linear):
                    if module.out_features >= 16:  # 至少 16 experts 才算路由器
                        parts = name.split(".")
                        # 从名字推断层编号
                        layer_nums = [p for p in parts if p.isdigit()]
                        idx = int(layer_nums[0]) if layer_nums else self._layer_count

                        def make_fallback_hook(i):
                            def fallback_hook(mod, inp, out):
                                with torch.no_grad():
                                    k = min(self.top_k, out.shape[-1])
                                    topk_idx = out.topk(k, dim=-1).indices
                                    self._raw[i].append(topk_idx.detach().cpu())
                            return fallback_hook

                        h = module.register_forward_hook(make_fallback_hook(idx))
                        self.hooks.append(h)
                        self._layer_count = max(self._layer_count, idx + 1)

    def reset(self):
        self._raw.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def compute_stats(self) -> Dict[str, any]:
        """
        返回每层 + 全局的负载统计。
        """
        if not self._raw:
            return {}

        layer_stats = {}
        all_entropies, all_cvs, all_imbalance = [], [], []

        for layer_idx in sorted(self._raw.keys()):
            indices_list = self._raw[layer_idx]
            if not indices_list:
                continue
            # 拼接所有 tokens 的专家选择：[total_tokens, top_k]
            all_idx = torch.cat(indices_list, dim=0)  # [N, top_k]
            flat = all_idx.view(-1).long()

            counts = torch.zeros(self.num_experts, dtype=torch.float)
            counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))

            total_selections = counts.sum().item()
            if total_selections == 0:
                continue

            prob = counts / total_selections

            # 熵
            prob_nonzero = prob[prob > 0]
            entropy = -(prob_nonzero * prob_nonzero.log()).sum().item()
            entropy_max = math.log(self.num_experts)
            entropy_norm = entropy / entropy_max

            # CV
            mean_load = counts.mean().item()
            std_load = counts.std().item()
            cv = std_load / (mean_load + 1e-9)

            # 不均衡比
            max_load = counts.max().item()
            imbalance_ratio = max_load / (mean_load + 1e-9)

            # 死专家（激活率 < 1% × 均值）
            dead_threshold = mean_load * 0.01
            dead_count = (counts < dead_threshold).sum().item()

            # 前10热专家
            top10 = counts.topk(10).indices.tolist()
            top10_fracs = (counts[top10] / total_selections * 100).tolist()

            layer_stats[layer_idx] = {
                "entropy_norm":    round(entropy_norm, 4),
                "cv":              round(cv, 4),
                "imbalance_ratio": round(imbalance_ratio, 4),
                "dead_experts":    int(dead_count),
                "top1_load_pct":   round(top10_fracs[0], 3),
                "top10_experts":   top10,
                "top10_load_pcts": [round(x, 3) for x in top10_fracs],
                "n_tokens":        int(total_selections // self.top_k),
            }

            all_entropies.append(entropy_norm)
            all_cvs.append(cv)
            all_imbalance.append(imbalance_ratio)

        if not all_entropies:
            return {"layer_stats": layer_stats}

        global_stats = {
            "num_moe_layers":    len(all_entropies),
            "entropy_norm_mean": round(float(np.mean(all_entropies)), 4),
            "entropy_norm_min":  round(float(np.min(all_entropies)), 4),
            "cv_mean":           round(float(np.mean(all_cvs)), 4),
            "cv_max":            round(float(np.max(all_cvs)), 4),
            "imbalance_mean":    round(float(np.mean(all_imbalance)), 4),
            "imbalance_max":     round(float(np.max(all_imbalance)), 4),
            # 预警：熵 < 0.8 或 CV > 0.5 或不均衡比 > 5
            "alert_low_entropy_layers":  [i for i, e in zip(sorted(self._raw.keys()), all_entropies) if e < 0.80],
            "alert_high_cv_layers":      [i for i, c in zip(sorted(self._raw.keys()), all_cvs) if c > 0.50],
            "alert_high_imbalance_layers": [i for i, r in zip(sorted(self._raw.keys()), all_imbalance) if r > 5.0],
        }

        return {"global": global_stats, "layer_stats": layer_stats}


# ─────────────────────────────────────────────────────────────────────────────
# 2. 训练回调（OpenRLHF / verl 集成）
# ─────────────────────────────────────────────────────────────────────────────

class MoELoadCallback:
    """
    轻量级训练回调，每 log_every_steps 步记录一次 MoE 负载。

    在 OpenRLHF 中集成:
        trainer = RewardModelTrainer(model=model, ...)
        moe_cb = MoELoadCallback(model, log_every_steps=50, output_dir=save_path)
        # 在训练循环中:
        #   moe_cb.on_step_end(global_step)

    在 verl 中集成:
        # 在 actor_model 的前向步骤前后调用 moe_cb.on_step_end(step)
    """

    def __init__(
        self,
        model: nn.Module,
        log_every_steps: int = 50,
        output_dir: str = "moe_logs",
        num_experts: int = 128,
        top_k: int = 8,
        rank: int = 0,          # DDP/FSDP rank，只在 rank=0 写文件
    ):
        self.log_every_steps = log_every_steps
        self.output_dir = Path(output_dir)
        self.rank = rank
        self._step = 0

        if rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = (self.output_dir / "moe_load_history.jsonl").open("a")

        self.collector = MoELoadCollector(model, num_experts=num_experts, top_k=top_k)
        self._history: List[dict] = []

    def on_step_end(self, global_step: int):
        self._step = global_step
        if global_step % self.log_every_steps != 0:
            self.collector.reset()
            return

        stats = self.collector.compute_stats()
        self.collector.reset()

        if not stats or self.rank != 0:
            return

        record = {"step": global_step, **stats.get("global", {})}
        self._history.append(record)
        self._log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._log_file.flush()

        g = stats.get("global", {})
        print(
            f"[MoE] step={global_step} "
            f"entropy_norm={g.get('entropy_norm_mean','?'):.3f} "
            f"(min={g.get('entropy_norm_min','?'):.3f}) "
            f"cv_mean={g.get('cv_mean','?'):.3f} "
            f"imbalance_max={g.get('imbalance_max','?'):.2f} "
            f"alerts_entropy={g.get('alert_low_entropy_layers',[])} "
            f"alerts_cv={g.get('alert_high_cv_layers',[])} "
        )

    def close(self):
        if self.rank == 0:
            self._log_file.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. 独立分析入口
# ─────────────────────────────────────────────────────────────────────────────

def load_model_for_analysis(model_path: str, device: str = "cuda"):
    """加载模型用于分析（4bit 量化节省显存）"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading model from {model_path} ...")
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        output_router_logits=True,   # 强制输出路由 logits（覆盖 config.json）
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def run_inference_for_analysis(
    model,
    tokenizer,
    data_file: str,
    max_samples: int = 200,
    max_len: int = 2048,
    batch_size: int = 4,
):
    """对 eval 数据集跑推理，触发 hook 收集专家激活"""
    import json

    records = []
    with open(data_file) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            r = json.loads(line)
            # 支持 openrlhf_rm 格式（prompt 是消息列表）
            if isinstance(r.get("prompt"), list):
                messages = r["prompt"]
            else:
                messages = [{"role": "user", "content": str(r.get("prompt", ""))}]
            records.append(messages)

    print(f"Running inference on {len(records)} samples ...")
    model.eval()
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            texts = [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                for msgs in batch
            ]
            inputs = tokenizer(
                texts, return_tensors="pt", truncation=True,
                max_length=max_len, padding=True
            ).to(model.device)
            # 只做一步前向，不生成（目的是触发 hook）
            _ = model(**inputs)
            if (i // batch_size) % 10 == 0:
                print(f"  {i + len(batch)}/{len(records)} samples processed")


def main():
    parser = argparse.ArgumentParser(description="MoE 专家负载分析")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_file",  required=True)
    parser.add_argument("--output_dir", default="moe_analysis")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--num_experts", type=int, default=128)
    parser.add_argument("--top_k",       type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_for_analysis(args.model_path)

    collector = MoELoadCollector(model, num_experts=args.num_experts, top_k=args.top_k)
    print(f"Registered hooks on {collector._layer_count} MoE layers")

    run_inference_for_analysis(
        model, tokenizer, args.data_file,
        max_samples=args.max_samples, batch_size=args.batch_size
    )

    stats = collector.compute_stats()
    collector.remove_hooks()

    # 保存详细 JSON
    out_json = output_dir / "moe_load_stats.json"
    with out_json.open("w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed stats saved to: {out_json}")

    # 打印全局摘要
    g = stats.get("global", {})
    print("\n" + "="*60)
    print("MoE 负载均衡分析结果")
    print("="*60)
    print(f"  分析 MoE 层数     : {g.get('num_moe_layers')}")
    print(f"  熵均值 (目标≥0.90): {g.get('entropy_norm_mean'):.4f}")
    print(f"  熵最小值 (预警<0.80): {g.get('entropy_norm_min'):.4f}")
    print(f"  CV均值 (目标≤0.20): {g.get('cv_mean'):.4f}")
    print(f"  CV最大值 (预警>0.50): {g.get('cv_max'):.4f}")
    print(f"  不均衡比均值       : {g.get('imbalance_mean'):.4f}")
    print(f"  不均衡比最大       : {g.get('imbalance_max'):.4f}")
    print(f"  低熵预警层         : {g.get('alert_low_entropy_layers', [])}")
    print(f"  高CV预警层         : {g.get('alert_high_cv_layers', [])}")

    # 生成可视化
    _plot_load_stats(stats, output_dir)


def _plot_load_stats(stats: dict, output_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    layer_stats = stats.get("layer_stats", {})
    if not layer_stats:
        return

    layers = sorted(layer_stats.keys())
    entropies  = [layer_stats[l]["entropy_norm"]    for l in layers]
    cvs        = [layer_stats[l]["cv"]              for l in layers]
    imbalances = [layer_stats[l]["imbalance_ratio"] for l in layers]
    dead_counts = [layer_stats[l]["dead_experts"]   for l in layers]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Qwen3-30B-MoE Expert Load Balance", fontsize=14)

    ax = axes[0, 0]
    ax.plot(layers, entropies, "b-o", markersize=3)
    ax.axhline(0.90, color="g", linestyle="--", label="Target ≥ 0.90")
    ax.axhline(0.80, color="r", linestyle="--", label="Alert < 0.80")
    ax.set_xlabel("Layer"); ax.set_ylabel("Normalized Entropy")
    ax.set_title("Per-layer Expert Load Entropy"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(layers, cvs, "r-o", markersize=3)
    ax.axhline(0.20, color="g", linestyle="--", label="Target ≤ 0.20")
    ax.axhline(0.50, color="r", linestyle="--", label="Alert > 0.50")
    ax.set_xlabel("Layer"); ax.set_ylabel("CV (std/mean)")
    ax.set_title("Per-layer Load CV"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(layers, imbalances, "g-o", markersize=3)
    ax.axhline(5.0, color="r", linestyle="--", label="Alert > 5.0")
    ax.set_xlabel("Layer"); ax.set_ylabel("max_load / mean_load")
    ax.set_title("Per-layer Imbalance Ratio"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.bar(layers, dead_counts, color="orange", alpha=0.7)
    ax.set_xlabel("Layer"); ax.set_ylabel("Dead Expert Count")
    ax.set_title("Dead Experts per Layer (< 1% × mean)"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "moe_load_per_layer.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {out_path}")

    # 选取最差层绘制专家分布条形图
    worst_layer = min(layers, key=lambda l: layer_stats[l]["entropy_norm"])
    ls = layer_stats[worst_layer]
    print(f"\n最差层 Layer {worst_layer}: entropy_norm={ls['entropy_norm']:.4f}")
    print(f"  Top-10 专家: {ls['top10_experts']}")
    print(f"  Top-10 激活率: {ls['top10_load_pcts']}%")


if __name__ == "__main__":
    main()
