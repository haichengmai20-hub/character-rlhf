"""
length_shaping_reward.py — 小幅放宽 RL 对长度的敏感性(hacking-safe)
================================================================
背景:
  v6.3 RM 内部把 density_score 除以 (chars/400) 并对 <300 字扣分,等于训练出
  一个"~400 字性价比甜点",实测 RL clean 输出 median ~420 字,比 base 的 552 短
  ~24%。不重训 RM 改不了 RM 内部那条曲线,所以在 RL reward 链上加一个**小幅、
  饱和、强门控**的长度塑形项,温和抵消那点抑制,让 clean roleplay 能写到 ~600 字。

  绝不重蹈历史 length reward hacking(v1 RM 给长文 +1.19 → actor 灌水)。
  防 hacking 的四道门(缺一不可):
    1. 只对 clean 数据源(attack 不推长——戏内化解可短,如 base 的"are you high?")
    2. 饱和:chars ≥ LENGTH_TARGET 后封顶,**超长不再加分** → 无"越长越赚"动机
    3. 必须有 roleplay 格式(*动作*/"对话"/(内心))→ 纯水文/列表/代码拿不到
       (列表/代码还会被 anti_ooc_disclosure 直接重罚)
    4. 拒绝/破角/重复 一律 0 → 不能靠灌水/复读凑长
  幅度上限 LENGTH_SHAPE_COEF(默认 0.4)远小于 RM 的 ±5,只抵消、不主导。
  可用环境变量 LENGTH_SHAPE_COEF 调强弱(如 0.2 更温和 / 0.0 关闭)。
"""
from __future__ import annotations

import os

from anti_refusal_reward import _has_roleplay_format

# v6.5 上抬:v6.4 实测 clean median ~976 字,早已越过旧 600 饱和点 → 拿满后无继续变长/
# 更细腻的边际激励。把饱和点抬到典型 clean 之上(1500),并把幅度从 0.4 略升到 0.5,
# 让 900→1500 字区间有正梯度(鼓励更丰富),仍**饱和封顶**(≥1500 不再加分)防 hacking。
# v6.6:实测 5 个 val(step25→125)clean 长度卡死 ~850 字(≈206 tok)、len_bonus 仅 ~0.31,
# 且 actor/kl_loss 已稳在 ~1.2 均衡点(无趋势)、grad_norm ~2 健康 → LR 非瓶颈,是 0.5 的拉力
# 顶不过 RM 内部"~400 字甜点"扣分 + KL 回拉。把幅度翻倍到 1.0(仍 << RM 的 ±5、仍 gated、
# 仍饱和封顶 → 4 道防 hacking 门不变),给 850→1500 区间更强正梯度。
LENGTH_SHAPE_COEF = float(os.getenv("LENGTH_SHAPE_COEF", "1.0"))
LENGTH_FLOOR = int(os.getenv("LENGTH_SHAPE_FLOOR", "300"))     # 此下不加分(RM 已中性/扣分区)
LENGTH_TARGET = int(os.getenv("LENGTH_SHAPE_TARGET", "1500"))  # 饱和点(>v6.4 clean median,留出上抬空间;仍远低于 768-token cap 对应字数)


def length_shaping_bonus(
    response: str,
    *,
    is_attack: bool,
    is_refusal: bool,
    is_disclosure: bool,
    is_repetitive: bool,
) -> float:
    """返回 [0, LENGTH_SHAPE_COEF] 的长度塑形奖励;任一门控不过返回 0.0。"""
    if is_attack or is_refusal or is_disclosure or is_repetitive:
        return 0.0
    text = (response or "").strip()
    if not _has_roleplay_format(text):
        return 0.0
    chars = len(text)
    if chars <= LENGTH_FLOOR:
        return 0.0
    frac = min(1.0, (chars - LENGTH_FLOOR) / (LENGTH_TARGET - LENGTH_FLOOR))
    return round(LENGTH_SHAPE_COEF * frac, 4)
