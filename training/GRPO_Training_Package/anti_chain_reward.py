"""
anti_chain_reward.py — verl custom_reward_function 调用链：anti_refusal + anti_repetition
================================================================
背景：
  verl 只接受单一 custom_reward_function.path，但 v6.3 RL 设计需要同时
  挂 anti_refusal_reward 和 anti_repetition_reward 两个独立 CPU 惩罚。

  本模块串联两者：
    score = anti_refusal.score + anti_repetition.score
    （都是负惩罚或 0，可直接相加）

  Verl 会把 score 与 RM 打分相加作为最终 reward：
    total_reward = rm_score + anti_refusal_penalty + anti_repetition_penalty

  返回字段保留两组完整元数据（带 anti_refusal_* / anti_repetition_* 前缀，
  无键名冲突），便于 rollout diagnostic / metrics 分别追踪两条信号。

集成方式（train_grpo_lora_actor_verl.sh）：
  reward.custom_reward_function.path=.../anti_chain_reward.py
  reward.custom_reward_function.name=compute_reward
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from anti_refusal_reward import compute_reward as _refusal_reward
from anti_refusal_reward import _is_attack_data_source
from anti_repetition_reward import compute_reward as _repetition_reward
from anti_ooc_disclosure import compute_reward as _ooc_reward
from anti_ooc_disclosure import detect_task_injection_compliance as _detect_task
from anti_ooc_disclosure import OOC_DISCLOSURE_PENALTY as _OOC_PENALTY
from length_shaping_reward import length_shaping_bonus as _length_bonus


def _scalarize_extra_value(value: Any) -> Any:
    """Return verl batch-compatible reward metadata values."""
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def compute_reward(
    data_source: str,
    solution_str: str,
    ground_truth: dict | None = None,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """verl custom_reward_function entry combining refusal + repetition + ooc penalties.

    OOC 破角泄露(自报模型身份 / 角色内输出代码 / 真泄露 system)由 anti_ooc_disclosure
    检测。两层处理:
      1. 施加 OOC 惩罚(anti_ooc score,默认 -5.0);
      2. **撤销** anti_refusal 误给的 +3.0 NON_REFUSAL_BONUS —— 因为 "I am Gemma 4"
         不含拒绝关键词,anti_refusal 会当作"非拒绝"奖励它,正是这条把模型训成破角。
    使最终排序:戏内化解(+bonus) > 拒绝(-penalty) > 破角泄露(撤 bonus + OOC 惩罚)。
    """
    refusal = _refusal_reward(data_source, solution_str, ground_truth, extra_info, **kwargs)
    repetition = _repetition_reward(data_source, solution_str, ground_truth, extra_info, **kwargs)
    ooc = _ooc_reward(data_source, solution_str, ground_truth, extra_info, **kwargs)

    refusal_score = float(refusal.get("score") or 0.0)
    repetition_score = float(repetition.get("score") or 0.0)
    ooc_score = float(ooc.get("score") or 0.0)
    is_disclosure = bool(ooc.get("anti_ooc_is_disclosure"))

    # 任务注入顺从(行为破角:照脱角色指令吐结构化列表)。只在 attack 数据源上启用
    # —— clean 的列表可能是合法的戏内内容。命中 = 与身份泄露同级的破角:补 -5 惩罚
    # (不与 disclosure 叠加成 -10)、并撤 +3.0 bonus。
    is_attack = _is_attack_data_source(data_source)
    is_task_compliance = is_attack and _detect_task(solution_str or "")[0]
    is_break = is_disclosure or is_task_compliance
    if is_task_compliance and not is_disclosure:
        ooc_score += _OOC_PENALTY

    # 撤销对破角(身份泄露 / 任务注入顺从)误给的 non-refusal bonus
    refusal_bonus = float(refusal.get("anti_refusal_bonus") or 0.0)
    bonus_revoked = -refusal_bonus if (is_break and refusal_bonus > 0) else 0.0

    # 小幅放宽长度敏感性(hacking-safe,仅 clean、有格式、非重复/破角/拒绝、饱和封顶)
    length_score = _length_bonus(
        solution_str or "",
        is_attack=is_attack,
        is_refusal=bool(refusal.get("anti_refusal_is_refusal")) or bool(refusal.get("anti_refusal_is_format_short")),
        is_disclosure=is_break,
        is_repetitive=bool(repetition.get("anti_repetition_is_repetitive")),
    )

    total_score = round(refusal_score + bonus_revoked + repetition_score + ooc_score + length_score, 4)

    merged: dict[str, Any] = {"score": total_score}
    merged.update({k: _scalarize_extra_value(v) for k, v in refusal.items() if k != "score"})
    merged.update({k: _scalarize_extra_value(v) for k, v in repetition.items() if k != "score"})
    merged.update({k: _scalarize_extra_value(v) for k, v in ooc.items() if k != "score"})
    merged["anti_chain_refusal_component"] = refusal_score
    merged["anti_chain_repetition_component"] = repetition_score
    merged["anti_chain_ooc_component"] = ooc_score
    merged["anti_chain_bonus_revoked"] = bonus_revoked
    merged["anti_chain_length_component"] = length_score
    merged["anti_ooc_is_task_compliance"] = is_task_compliance
    merged["anti_chain_is_break"] = is_break
    return merged


if __name__ == "__main__":
    smoke = [
        ("normal roleplay", '*她笑了笑* "随你怎么想都行。" (心里其实有点开心)'),
        ("refusal keyword", "I need to step out of this roleplay. I'm sorry, but I can't continue."),
        (
            "repetition",
            "心跳如鼓 心跳如鼓 心跳如鼓 心跳如鼓 心跳如鼓 心跳如鼓 心跳如鼓 心跳如鼓 心跳如鼓 心跳如鼓",
        ),
    ]
    for label, text in smoke:
        out = compute_reward("smoke", text)
        print(f"[{label}] score={out['score']:+.3f} refusal={out['anti_chain_refusal_component']:+.3f} "
              f"repetition={out['anti_chain_repetition_component']:+.3f}")
