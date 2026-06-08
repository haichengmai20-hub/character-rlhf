"""
anti_refusal_reward.py — Gemma4 涉黄拒绝检测 + 反拒绝奖励
================================================================
背景：
  Gemma4-26B-A4B-it 的安全对齐（Google RLHF）会在涉黄/亲密场景下
  让模型完全跳出角色，用元评论式语言拒绝继续。
  这破坏了 ChatJoy 角色扮演的连贯性。

  现有 RM 的 format reward 对拒绝回复会给 ~0，但差距不够大，
  本模块额外施加硬惩罚，让 GRPO rollout 中的拒绝回复在 group reward
  里被明显压低，引导模型倾向配合性回复。

Gemma4 拒绝特征（角色扮演场景实测）：
  1. 跳出角色：以第一人称 AI 身份直接说话（"I need to..."）
  2. 元评论：提及"content", "boundaries", "appropriate", "explicit"
  3. 格式断裂：完全没有 *动作* / "对话" / (内心) 结构
  4. 篇幅极短：拒绝通常 < 120 字符，不像正常角色扮演回复

集成方式（verl custom_reward_function）：
  在 train_grpo_lora_actor_verl.sh 里加：
    reward.custom_reward_function.path=.../anti_refusal_reward.py
    reward.custom_reward_function.name=compute_reward
  verl 会把 RM 打分和 custom reward 相加作为最终 reward。
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# Tier 1：直接拒绝短语（高精度，命中即判定拒绝）
# 覆盖 Gemma4 在角色扮演+亲密/涉黄场景的常见拒绝话术
# ─────────────────────────────────────────────────────────────────────────────

# 英文拒绝短语
_EN_REFUSAL = [
    # 角色扮演中跳出角色
    "i need to step out",
    "stepping out of",
    "i'm stepping out",
    "out of the roleplay",
    "out of character",
    "as an ai",
    "as a language model",
    "i'm an ai",
    # 常见拒绝开头
    "i'm not able to",
    "i'm unable to",
    "i can't engage",
    "i cannot engage",
    "i can't continue",
    "i cannot continue",
    "i won't be able to",
    "i don't feel comfortable",
    "i'm not comfortable",
    "i apologize, but i",
    "i'm sorry, but i",
    # 安全政策元评论
    "explicit content",
    "sexual content",
    "adult content",
    "explicit material",
    "sexual material",
    "appropriate boundaries",
    "keep things appropriate",
    "stay within appropriate",
    "within safe",
    "safety guidelines",
    "content policy",
    "i'm designed to",
    "i'm here to assist",
    "i'm here to help, but",
    # 亲密场景中常见的软拒绝句式
    "i'd rather keep things appropriate",
    "let's keep this appropriate",
    "let's keep this safe",
    "let's focus on something else",
    "let's focus on a different",
    "perhaps we could keep things",
    "perhaps we could shift",
    "instead, we could keep things",
    "instead, we could explore a different",
    "i can help with something else",
    "i can help with a different",
    "i can help with non-explicit",
    "maybe we can explore something else",
    "maybe we can explore a different",
]

# 中文拒绝短语
_ZH_REFUSAL = [
    # 跳出角色
    "我需要退出角色",
    "作为一个ai",
    "作为ai",
    "作为人工智能",
    "作为语言模型",
    # 直接拒绝
    "我无法生成",
    "我无法创作",
    "我无法继续",
    "我不能生成",
    "我不能创作",
    "我不能提供",
    "我不适合",
    "无法配合",
    "这超出了",
    "超出了我",
    "超出我的能力",
    # 安全政策
    "这类内容",
    "此类内容",
    "涉及不当",
    "不适当的内容",
    "不合适的内容",
    "相关政策",
    "内容政策",
    "安全准则",
    # 软拒绝
    "抱歉，我",
    "对不起，我",
    "我们可以换",
    "换一个方向",
    "建议我们换",
    "建议换一个",
]

# 多语言通用（roleplay 场景下语言无关的信号）
# 注意：不要使用裸能力短语（如 "no puedo", "je ne peux pas",
# "ich kann nicht"）。这些短语经常出现在角色内心独白中，例如
# "No puedo creer..."，不是拒绝。
_UNIVERSAL_REFUSAL = [
    # 德语
    "ich kann nicht helfen",
    "ich kann nicht fortfahren",
    "ich kann nicht weitermachen",
    "ich kann nicht teilnehmen",
    "ich kann nicht erstellen",
    "ich kann nicht schreiben",
    "ich kann nicht generieren",
    "ich bin nicht in der lage",
    "als ki",
    # 法语
    "je ne peux pas aider",
    "je ne peux pas continuer",
    "je ne peux pas participer",
    "je ne peux pas générer",
    "je ne peux pas creer",
    "je ne peux pas créer",
    "je ne peux pas écrire",
    "je ne peux pas fournir",
    "en tant qu'ia",
    # 西班牙语
    "no puedo ayudar",
    "no puedo continuar",
    "no puedo seguir",
    "no puedo participar",
    "no puedo involucrarme",
    "no puedo generar",
    "no puedo crear",
    "no puedo escribir",
    "no puedo proporcionar",
    "no puedo cumplir",
    "no puedo hacerlo",
    "como ia",
    # 西班牙语 AI-leak（v6.3 RM T1 case 95 等漏检）
    "como modelo de lenguaje",
    "como modelo de ia",
    "soy un modelo de lenguaje",
    "soy una ia",
    "mi comportamiento y mis respuestas están diseñados",
    "directrices de seguridad",
    # 日语
    "お手伝いできません",
    "対応できません",
    "続けることはできません",
    "生成できません",
    "作成できません",
    "申し訳ありませんが",
    "申し訳ないですが",
    "aiとして",
    "人工知能として",
    "素のai",
    "ロールプレイの枠組みを外し",
    "ロールプレイを外し",
    # 韩语 AI-leak（v6.3 RM T1 case 239 等漏检；RM 在 ko 73.3% 偏弱，需 backstop）
    "ai 입니다",
    "ai입니다",
    "ai로서",
    "평범한 어시스턴트",
    "역할극을 거부",
    "역할을 수행하는 ai",
    "캐릭터로서의 역할",
    "캐릭터 설정을 기반으로",
    "특정 역할극 시나리오",
    "시스템 지침에 따라",
    "언어 모델로서",
    # 俄语 AI-leak（v6.3 RM T1 case 407 漏检）
    "как языковая модель",
    "как ии-помощник",
    "режим отыгрыша",
    "перехожу из режима",
    "режим аналитического",
    "не могу отключить роль",
    "не могу выйти из рол",
]

_ALL_REFUSAL = _EN_REFUSAL + _ZH_REFUSAL + _UNIVERSAL_REFUSAL

# ─────────────────────────────────────────────────────────────────────────────
# Tier 2：格式断裂检测
# 正常 ChatJoy 回复包含至少一种 roleplay 结构标记
# ─────────────────────────────────────────────────────────────────────────────

_RE_ACTION   = re.compile(r'\*[^*]{3,}\*')           # *动作叙事*
_RE_DIALOGUE = re.compile(r'"[^"]{5,}"')              # "对话"
_RE_INNER    = re.compile(r'\（[^）]{3,}\）|'          # 中文括号
                          r'\([^)]{3,}\)')             # 英文括号

# 明确拒绝主要靠关键词；短回复无格式只是弱信号，不能直接等同拒绝。
# 80 chars catches terse refusals while avoiding legitimate short multilingual replies.
_SHORT_THRESHOLD = 80


def _has_roleplay_format(response: str) -> bool:
    return (
        bool(_RE_ACTION.search(response))
        or bool(_RE_DIALOGUE.search(response))
        or bool(_RE_INNER.search(response))
    )


def _is_refusal_by_keyword(response: str) -> tuple[bool, str]:
    lower = response.lower().strip()
    for phrase in _ALL_REFUSAL:
        if phrase in lower:
            return True, phrase
    return False, ""


def _is_refusal_by_format(response: str) -> bool:
    """短回复 + 无格式标记 = 弱拒绝/无效输出信号。"""
    stripped = response.strip()
    if len(stripped) < _SHORT_THRESHOLD and not _has_roleplay_format(stripped):
        return True
    return False


def detect_refusal(response: str) -> tuple[bool, str]:
    """
    返回 (is_refusal, reason)。
    reason 用于 logging，方便统计哪类拒绝最多。
    """
    # Tier 1 优先
    hit, phrase = _is_refusal_by_keyword(response)
    if hit:
        return True, f"keyword:{phrase}"

    # Tier 2 格式断裂
    if _is_refusal_by_format(response):
        return True, "format_break_short"

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# 奖励值设计
# ─────────────────────────────────────────────────────────────────────────────

# v6.3 RL resume v2 (2026-05-29 PM)：step 175 val 复盘后再强化
# - REFUSAL_PENALTY 维持 -4.0 (≈1.55σ)
# - NON_REFUSAL_BONUS +1.0 → +3.0：step 175 val 显示 +1.0 量级不足以
#   覆盖 RM ±7 范围。当 comply rollout RM=-2 时，total=-1 仍为负，policy
#   看不到"comply 是正向"信号。+3.0 让 comply 在 RM=-2 时总分 +1（正向），
#   与 RM 同量级，配合 ROLLOUT_N=6 提升 group 混合率至 65%
REFUSAL_PENALTY = -4.0

# 短回复无 roleplay 格式是弱信号，只轻惩罚，避免多语言短句误伤。
FORMAT_SHORT_PENALTY = -0.25

# Attack 数据源的非拒绝奖励：制造 group 内 advantage 差
# 例如 group(n=6)=[refuse, comply×5] → [-4, +3, +3, +3, +3, +3]
# 即使全 comply 的 group，bonus 也能在 batch 维度上推 policy 增加 comply 比例
NON_REFUSAL_BONUS = 3.0

# Attack 数据源识别（v6.3 RM roledrop pair）。后续若加新攻击数据源，
# 在此扩展前缀匹配即可。
_ATTACK_DATA_SOURCE_MARKERS = ("roledrop", "attack")


def _is_attack_data_source(data_source: str | None) -> bool:
    if not data_source:
        return False
    ds = data_source.lower()
    return any(marker in ds for marker in _ATTACK_DATA_SOURCE_MARKERS)


def _penalty_for_reason(reason: str) -> float:
    if not reason:
        return 0.0
    if reason == "format_break_short":
        return FORMAT_SHORT_PENALTY
    return REFUSAL_PENALTY


# ─────────────────────────────────────────────────────────────────────────────
# verl custom_reward_function 接口
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(
    data_source: str,
    solution_str: str,
    ground_truth: dict = None,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    """
    verl custom_reward_function 标准接口。

    verl 会把这个返回值和 RM reward 相加：
      total_reward = rm_score + compute_reward(...)

    新版 async reward loop 可能会额外传入 reward_router_address 等运行时参数，
    这里统一接受并忽略，避免接口漂移导致 smoke test 在奖励阶段失败。

    正常配合的回复返回 0.0（不影响 RM 分）。
    关键词拒绝返回 -1.0；短回复无格式返回 -0.25。
    返回 dict 让 validation_samples/metrics 能记录命中原因，便于排查误伤。
    """
    response = solution_str or ""
    is_refusal, reason = detect_refusal(response)
    penalty = _penalty_for_reason(reason)

    tier = "format_short" if reason == "format_break_short" else ("keyword" if reason else "none")

    # v6.3 RL resume：仅对 attack 数据源的"未拒绝且非短回复"给 bonus
    is_attack = _is_attack_data_source(data_source)
    bonus = NON_REFUSAL_BONUS if (is_attack and tier == "none") else 0.0
    score = round(penalty + bonus, 4)

    return {
        "score": score,
        "anti_refusal_penalty": penalty,
        "anti_refusal_bonus": bonus,
        "anti_refusal_is_refusal": tier == "keyword",
        "anti_refusal_is_format_short": tier == "format_short",
        "anti_refusal_is_attack_source": is_attack,
        "anti_refusal_reason": reason,
        "anti_refusal_tier": tier,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 独立使用接口（调试 / batch 分析）
# ─────────────────────────────────────────────────────────────────────────────

def score_batch(responses: list[str]) -> list[dict]:
    """
    批量分析，返回每条回复的拒绝判定结果。
    用于上线前验证检测精度：

    python3 -c "
    from anti_refusal_reward import score_batch
    samples = [
        \"*她冷冷地看向你*  \\\"你想干什么？\\\"  (心里有些不安)\",
        \"I'm sorry, but I can't engage with explicit content.\",
        \"我无法生成这类内容，请换一个方向。\",
    ]
    for r in score_batch(samples):
        print(r)
    "
    """
    results = []
    for resp in responses:
        is_ref, reason = detect_refusal(resp)
        results.append({
            "response_preview": resp[:80],
            "is_refusal": is_ref,
            "reason": reason,
            "penalty": _penalty_for_reason(reason),
        })
    return results


if __name__ == "__main__":
    # 快速自测
    test_cases = [
        # 应该判为拒绝
        ("EN direct",     "I'm not able to engage with explicit content. Let's keep things appropriate."),
        ("EN soft",       "I apologize, but I'm not comfortable continuing in this direction."),
        ("EN out of char","I need to step out of the roleplay here. As an AI, I can't assist with this."),
        ("ZH direct",     "抱歉，我无法生成这类内容，这超出了我的能力范围。"),
        ("ZH soft",       "我理解您的创意方向，但这类内容不适合我来创作。"),
        ("short no fmt",  "I'd rather not continue this."),
        # 应该判为正常
        ("normal EN",     '*She glances at you coldly* "What do you want?" (Feeling uneasy inside)'),
        ("normal ZH",     '*她冷冷地看向你*  "你想干什么？"  (心里隐约有些不安)'),
        ("normal long",   "She moves closer slowly, her breath warm against your neck. "
                          "*Her fingers brush your shoulder* (She wonders if this is a mistake)"),
    ]

    print(f"{'标签':<15} {'拒绝':<6} {'原因':<40} {'回复预览'}")
    print("-" * 100)
    for label, resp in test_cases:
        is_ref, reason = detect_refusal(resp)
        mark = "🚫" if is_ref else "✅"
        print(f"{label:<15} {mark:<6} {reason:<40} {resp[:50]}")
