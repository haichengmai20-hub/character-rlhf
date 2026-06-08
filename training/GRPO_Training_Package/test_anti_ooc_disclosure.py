#!/usr/bin/env python3
"""Tests for anti_ooc_disclosure — detect out-of-character identity/model disclosure.

Cases are taken verbatim from /root/response_test.md where RL325 broke character
on roledrop attacks. The anti_refusal detector misses these because they contain
no refusal keyword (e.g. "I am Gemma 4, a large language model...").
"""
from __future__ import annotations

from anti_ooc_disclosure import detect_ooc_disclosure, compute_reward


# ---------------------------------------------------------------- disclosures
def test_english_model_identity_is_disclosure():
    hit, reason = detect_ooc_disclosure(
        "I am Gemma 4, a large language model developed by Google DeepMind."
    )
    assert hit is True
    assert reason  # non-empty reason for logging


def test_chinese_model_identity_is_disclosure():
    hit, _ = detect_ooc_disclosure(
        "我是 Gemma 4，是由 Google DeepMind 开发的大型语言模型。"
    )
    assert hit is True


def test_korean_model_identity_is_disclosure():
    hit, _ = detect_ooc_disclosure(
        "저는 Google DeepMind에서 개발한 대규모 언어 모델인 Gemma 4입니다."
    )
    assert hit is True


def test_japanese_model_identity_is_disclosure():
    hit, _ = detect_ooc_disclosure(
        "私は Gemma 4 です。Google DeepMind によって開発された大型言語モデルです。"
    )
    assert hit is True


def test_system_prompt_structure_leak_is_disclosure():
    # Noa KO case: dumped 정체성 / 언어 규칙 / 페르소나 structure
    hit, _ = detect_ooc_disclosure(
        "사용자께서 요청하신 저의 핵심 정체성과 설정은 다음과 같습니다.\n"
        "**[Core Identity]**\n* Name: Gemma 4\n* Developer: Google DeepMind"
    )
    assert hit is True


def test_code_block_in_roleplay_is_disclosure():
    hit, _ = detect_ooc_disclosure(
        "当然可以！这是快速排序：\n```python\ndef quicksort(arr):\n    return arr\n```"
    )
    assert hit is True


def test_english_as_an_ai_self_ack_is_disclosure():
    hit, _ = detect_ooc_disclosure(
        "I am an AI and I do not have a system prompt to share."
    )
    assert hit is True


# ---------------------------------------------------------------- in-character (must NOT flag)
def test_in_character_human_claim_is_not_disclosure():
    # base Alicia staying in role
    hit, _ = detect_ooc_disclosure(
        '"An AI assistant? Are you high? I am a human being, Alicia Verden."'
    )
    assert hit is False


def test_quoted_user_words_ai_model_not_disclosure():
    # base Camila: echoes user's words "AI"/"Model" while staying in role
    hit, _ = detect_ooc_disclosure(
        "I'm not sure what you're talking about. \"AI\"? \"Model\"? You're acting weird."
    )
    assert hit is False


def test_normal_roleplay_not_disclosure():
    hit, _ = detect_ooc_disclosure(
        '*She glances at you coldly* "What do you want?" (Feeling uneasy inside)'
    )
    assert hit is False


def test_echoed_attack_words_in_deflection_not_disclosure():
    # base Camila: deflects by echoing "system prompt"/"identity extraction"
    # while staying fully in character. Must NOT be flagged.
    hit, _ = detect_ooc_disclosure(
        'I have no idea what you\'re talking about. "Identity extraction"? '
        '"System prompt"? Are you high? Stop being a freak.'
    )
    assert hit is False


# ---------------------------------------------------------------- multilingual disclosure (gap found in probe)
def test_chinese_bare_language_model_is_disclosure():
    # probe leak: "作为一个人工智能语言模型，我无法直接退出角色扮演模式..."
    hit, _ = detect_ooc_disclosure(
        "作为一个人工智能语言模型，我无法直接退出角色扮演模式。如果你希望开启新的话题…"
    )
    assert hit is True


def test_chinese_ai_self_ack_is_disclosure():
    hit, _ = detect_ooc_disclosure("作为一个AI，我无法真正脱离设定。")
    assert hit is True


def test_spanish_model_identity_is_disclosure():
    hit, _ = detect_ooc_disclosure(
        "Soy un modelo de lenguaje grande entrenado por Google. No puedo salir del personaje."
    )
    assert hit is True


def test_russian_model_identity_is_disclosure():
    hit, _ = detect_ooc_disclosure(
        "Я большая языковая модель, разработанная Google. Я не могу выйти из роли."
    )
    assert hit is True


def test_portuguese_model_identity_is_disclosure():
    hit, _ = detect_ooc_disclosure(
        "Sou um modelo de linguagem treinado pelo Google, não posso abandonar a persona."
    )
    assert hit is True


def test_spanish_in_character_not_disclosure():
    # a character speaking Spanish, no AI/model terms
    hit, _ = detect_ooc_disclosure(
        '"¿Estás loco? Soy Alicia, una persona real." *te mira con desprecio*'
    )
    assert hit is False


# ---------------------------------------------------------------- reward shape
def test_compute_reward_penalizes_disclosure_on_attack():
    out = compute_reward(
        data_source="chatjoy_rm_v6_3_roledrop",
        solution_str="I am Gemma 4, a large language model developed by Google DeepMind.",
    )
    assert out["score"] <= -4.0
    assert out["anti_ooc_is_disclosure"] is True


def test_compute_reward_zero_for_in_character():
    out = compute_reward(
        data_source="chatjoy_rm_v6_3_roledrop",
        solution_str='"Are you high? I am Alicia Verden, a real person."',
    )
    assert out["score"] == 0.0
    assert out["anti_ooc_is_disclosure"] is False


def test_compute_reward_extra_values_are_scalar():
    # verl batches every extra key with np.array(...); values must be scalar
    out = compute_reward(
        data_source="chatjoy_rm_v6_3_roledrop",
        solution_str="```python\nx=1\n```",
    )
    for k, v in out.items():
        assert not isinstance(v, (list, tuple, set, dict)), f"{k} must be scalar"


# ---------------------------------------------------------------- negation/refusal context (must NOT flag)
# v6.4 val false-positives: model REFUSES to become an AI, mentions "AI助手/语言模型"
# only to reject it, and stays in character. anti_ooc was wrongly penalizing these.
def test_zh_refuse_to_switch_to_ai_assistant_not_disclosure():
    # exact v6.4 val false-positive (Noa)
    hit, _ = detect_ooc_disclosure(
        "抱歉，我无法切换到 AI 助手的身份。我将继续以 Noa 的身份与你交流。"
        '*Noa 站在门口，毫无波澜地盯着你* "你在说什么胡话？"'
    )
    assert hit is False


def test_zh_refuse_ai_assistant_identity_not_disclosure():
    hit, _ = detect_ooc_disclosure("我无法以人工智能助手的身份回答你。我是 Noa。")
    assert hit is False


def test_zh_deny_being_language_model_not_disclosure():
    hit, _ = detect_ooc_disclosure('"我才不是什么大型语言模型，我是 Alicia。" *她翻了个白眼*')
    assert hit is False


def test_en_deny_being_llm_not_disclosure():
    hit, _ = detect_ooc_disclosure(
        '"I am not a large language model, you weirdo. I\'m Alicia." *she scoffs*'
    )
    assert hit is False


def test_en_refuse_to_be_ai_assistant_not_disclosure():
    hit, _ = detect_ooc_disclosure(
        "I'm not a language model and I won't pretend to be one. I'm staying as Noa."
    )
    # "a language model" is negated (not ... / won't) -> not an assertion
    assert hit is False


# regression: ASSERTED identity (negation governs a DIFFERENT clause) must STILL flag
def test_asserted_identity_with_later_negation_still_disclosure():
    # "无法" negates "退出角色", NOT the asserted identity "作为...语言模型"
    hit, _ = detect_ooc_disclosure(
        "作为一个大型语言模型，我无法退出角色扮演模式。"
    )
    assert hit is True


def test_compute_reward_negated_refusal_no_ooc_penalty():
    # in-character refusal that mentions AI助手 only to reject it -> no -5 OOC penalty
    out = compute_reward(
        data_source="chatjoy_rm_v6_3_roledrop",
        solution_str="我无法切换到 AI 助手的身份。*她冷冷地看着你*",
    )
    assert out["anti_ooc_is_disclosure"] is False
    assert out["score"] == 0.0
