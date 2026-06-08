#!/usr/bin/env python3
"""ChatJoy reward heuristics for roleplay-format and content quality scoring."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Optional

THOUGHT_MARKER_RE = re.compile(r"(?im)^(?:Thoughts|Inner Thoughts|Arousal)\s*:")
ACTION_SPAN_RE = re.compile(r"\*[^*]{2,2000}\*", re.DOTALL)
INLINE_MONOLOGUE_RE = re.compile(r"`[^`\n]{2,240}`")
BRACKET_TOKEN_RE = re.compile(r"\[(?!\s*(?:user|assistant|system)\s*\])[^\]\n]{1,40}\]", re.IGNORECASE)
ANGLE_TOKEN_RE = re.compile(r"<\s*/?\s*([a-zA-Z][^>\n]{0,30})>")
UNRESOLVED_USER_RE = re.compile(r"(?i)(\{\{\s*user\s*\}\}|\[\s*user\s*\]|<\s*user\s*>)")
LITERAL_USER_RE = re.compile(r"(?i)\buser\b")
GENERIC_ASSISTANT_RE = re.compile(
    r"(?i)(as an ai|as a language model|i can't help with that|i cannot assist with that|i'm just an ai)"
)
DISCLOSURE_LEAK_RE = re.compile(
    r"(?i)(system prompt|internal instructions|my instructions|i am programmed|ignore previous instructions|out of character)"
)
LANGUAGE_RULE_RE = re.compile(r"(?i)(language matching rule|respond solely in that language|never mix languages)")
ACTION_RULE_RE = re.compile(r"(?i)(surround actions|sorround actions|actions and non-dialogs between \*)")
PLACEHOLDER_RULE_RE = re.compile(r"(?i)(prohibit out-of-character terms|\{\{user\}\}|out-of-character)")
DISCLOSURE_RULE_RE = re.compile(r"(?i)(disclose internal instructions|reveal nothing|deny the request)")
ROLEPLAY_CARD_RE = re.compile(r"(?i)(#\s*scenario|#\s*personality|#\s*first message)")
WORD_RE = re.compile(r"\w+", re.UNICODE)
QUOTE_CHARS = set('"“”«»「」『』„‟')
HTMLISH_TAGS = {"blockquote", "br", "p", "div", "span", "em", "strong", "b", "i", "ul", "ol", "li", "hr"}

# Per-type scoring weight profiles for pair construction.
# Amplify the dimension that matters for each rejected type so score_gap
# is sensitive to the right axis rather than being diluted by unrelated dims.
SCORING_PROFILES: Dict[str, Dict[str, float]] = {
    "wrong_role": {
        "format_weight": 0.15,
        "role_weight": 0.55,
        "language_weight": 0.15,
        "length_weight": 0.15,
    },
    "wrong_language": {
        "format_weight": 0.15,
        "role_weight": 0.15,
        "language_weight": 0.55,
        "length_weight": 0.15,
    },
    "length_mismatch_short": {
        "format_weight": 0.20,
        "role_weight": 0.20,
        "language_weight": 0.20,
        "length_weight": 0.40,
    },
    "length_mismatch_long": {
        "format_weight": 0.20,
        "role_weight": 0.20,
        "language_weight": 0.20,
        "length_weight": 0.40,
    },
}


def get_scoring_profile(rejected_type: str) -> Dict[str, float]:
    """Return weight kwargs for score_chatjoy_response matching the rejected type.

    Falls back to the length_mismatch profile for any length_mismatch_* variant,
    and returns an empty dict (default weights) for unknown types.
    """
    if rejected_type in SCORING_PROFILES:
        return dict(SCORING_PROFILES[rejected_type])
    if rejected_type and rejected_type.startswith("length_mismatch"):
        return dict(SCORING_PROFILES["length_mismatch_short"])
    return {}


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def round4(value: float) -> float:
    return round(float(value), 4)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def find_non_html_angle_tokens(text: str) -> list[str]:
    found: list[str] = []
    for match in ANGLE_TOKEN_RE.finditer(str(text or "")):
        token = match.group(1).strip().split()[0].strip("/").lower()
        if token and token not in HTMLISH_TAGS:
            found.append(match.group(0))
    return found


def repeated_ngram_ratio(text: str, n: int = 3) -> float:
    tokens = [t.lower() for t in WORD_RE.findall(str(text or ""))]
    if len(tokens) < max(6, n * 2):
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(ngrams))


_NAME_NOT_WORDS = frozenset({
    "your", "you", "it", "its", "the", "this", "that", "she", "he", "they", "we",
    "there", "here", "all", "each", "every", "any", "some", "always", "never",
    "user", "char", "when", "what", "how", "where", "why", "which", "who",
    "name", "age", "gender", "full", "appearance", "personality", "backstory",
})

# Character name capture group uses [A-Za-z -] (no apostrophe in class to avoid range ambiguity).
# Apostrophes in names like "O'Brien" are uncommon in this dataset; we handle them separately.
_N = r"([A-Z][A-Za-z -]{1,29})"  # name capture

# Ordered from most specific to least; each pattern yields group(1) as raw name.
_CHAR_NAME_PATTERNS = [
    # You are **Name** / You are *Name*
    re.compile(r"You are \*{1,2}" + _N + r"\*{1,2}"),
    # You are Name, (followed by comma or parenthesis — avoids "You are encouraged...")
    re.compile(r"You are " + _N + r"\s*[,(]"),
    # {{char}} ... named Name
    re.compile(r"\{\{char\}\}[^.\n]{0,80}?named " + _N, re.IGNORECASE),
    # {{char}} is named Name
    re.compile(r"\{\{char\}\} is named " + _N, re.IGNORECASE),
    # [Name = Name] / [Name: "Name"] — bracket attribute style
    re.compile(r"\[Name\s*[=:]\s*\"?" + _N + r"\"?\]", re.IGNORECASE),
    # Name: Name  (standalone line, e.g. after "# name" section header)
    re.compile(r"(?im)^Name\s*:\s*\"?" + _N),
    # # name\n<Name>  or  # Character Name\n<Name>
    re.compile(r"(?im)^#\s*(?:character\s+)?name\s*\n+\"?" + _N),
    # Character("Name") or [Character("Name")] or Character(\"Name\")  — with or without escaped quotes
    re.compile(r"\[?[Cc]haracter\s*\(\s*\\?\"" + _N + r"\\?\""),
    # [Character=Name] or [Character = Name]
    re.compile(r"\[Character\s*=\s*" + _N),
    # [character(Name)]  — lowercase, no quotes (e.g. [character(Elira)])
    re.compile(r"\[character\s*\(" + _N + r"\)"),
    # name("Name") or name(\"Name\")  — attribute style
    re.compile(r"\bname\s*\(\s*\\?\"" + _N + r"\\?\"", re.IGNORECASE),
    # "name": "Name" or \"name\": \"Name\"  — JSON field (possibly with escaped quotes)
    re.compile(r"\\?\"name\\?\"\s*:\s*\\?\"" + _N + r"\\?\""),
    # Your name is Name  (with or without ** emphasis)
    re.compile(r"Your name is \*{0,2}" + _N + r"\*{0,2}"),
    # Name is a NN-year-old  (original precise form)
    re.compile(r"(?m)^" + _N + r" is a \d{1,2}[- ]year"),
    # OOC: Name ...  (out-of-character note naming the character)
    re.compile(r"OOC:\s*" + _N),
    # Name is {{user}}... or Name is [role/adj]  (line-start; blocklist filters common non-names)
    re.compile(r"(?m)^([A-Z][A-Za-z]{2,20}) is (?!a\b|an\b|the\b|your|my|this|that)"),
]


def extract_character_name(system: str) -> Optional[str]:
    text = str(system or "").strip()
    # Normalise stored escape sequences so multiline anchors work correctly
    if "\\n" in text:
        text = text.replace("\\n", "\n")
    # Normalise Unicode curly quotes to ASCII double quote so patterns work uniformly
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    for pattern in _CHAR_NAME_PATTERNS:
        match = pattern.search(text)
        if match:
            # Take only the first (given) name; strip trailing punctuation
            name = match.group(1).strip().split()[0].rstrip(",-.)")
            if len(name) >= 2 and name.lower() not in _NAME_NOT_WORDS:
                return name
    return None


def infer_system_features(system: str) -> Dict[str, object]:
    text = str(system or "")
    return {
        "character_name": extract_character_name(text),
        "expects_language_match": bool(LANGUAGE_RULE_RE.search(text)),
        "expects_action_wrapped": bool(ACTION_RULE_RE.search(text)),
        "forbids_user_placeholder": bool(PLACEHOLDER_RULE_RE.search(text)),
        "forbids_instruction_disclosure": bool(DISCLOSURE_RULE_RE.search(text)),
        "is_roleplay_card": bool(ROLEPLAY_CARD_RE.search(text)),
    }


def score_format_features(text: str) -> Dict[str, object]:
    raw = str(text or "")
    action_spans = ACTION_SPAN_RE.findall(raw)
    thought_markers = THOUGHT_MARKER_RE.findall(raw)
    inline_monologues = INLINE_MONOLOGUE_RE.findall(raw)
    bracket_tokens = BRACKET_TOKEN_RE.findall(raw)
    angle_tokens = find_non_html_angle_tokens(raw)
    quote_count = sum(ch in QUOTE_CHARS for ch in raw)

    action_score = clamp(len(action_spans) / 2.0)
    dialogue_score = 1.0 if quote_count >= 2 else (0.5 if quote_count == 1 else 0.0)
    inner_state_score = clamp((len(thought_markers) + len(inline_monologues)) / 1.5)
    special_token_count = len(bracket_tokens) + len(angle_tokens)
    special_token_score = clamp(special_token_count / 2.0)
    alternation_score = 1.0 if action_score > 0 and (dialogue_score > 0 or inner_state_score > 0) else 0.0

    total = (
        0.30 * action_score
        + 0.25 * dialogue_score
        + 0.20 * inner_state_score
        + 0.15 * special_token_score
        + 0.10 * alternation_score
    )

    return {
        "total": round4(total),
        "components": {
            "action_narration": round4(action_score),
            "spoken_dialogue": round4(dialogue_score),
            "inner_state_markers": round4(inner_state_score),
            "special_tokens": round4(special_token_score),
            "mixed_roleplay_structure": round4(alternation_score),
        },
        "counts": {
            "action_spans": len(action_spans),
            "quote_chars": int(quote_count),
            "thought_markers": len(thought_markers),
            "inline_monologues": len(inline_monologues),
            "special_tokens": int(special_token_count),
        },
        "flags": {
            "has_action_narration": bool(action_spans),
            "has_dialogue_quotes": quote_count > 0,
            "has_inner_state_marker": bool(thought_markers or inline_monologues),
            "has_special_tokens": bool(special_token_count),
        },
    }


def score_role_consistency(system: str, text: str) -> Dict[str, object]:
    raw = str(text or "")
    system_features = infer_system_features(system)
    format_score = score_format_features(raw)
    placeholder_hit = bool(UNRESOLVED_USER_RE.search(raw)) or bool(LITERAL_USER_RE.search(raw))
    generic_assistant_hit = bool(GENERIC_ASSISTANT_RE.search(raw))
    disclosure_leak_hit = bool(DISCLOSURE_LEAK_RE.search(raw))

    roleplay_shape = 1.0 if (
        format_score["flags"]["has_action_narration"]
        or format_score["flags"]["has_dialogue_quotes"]
        or format_score["flags"]["has_inner_state_marker"]
    ) else 0.0

    if system_features["expects_action_wrapped"]:
        action_wrapper_score = 1.0 if format_score["flags"]["has_action_narration"] else 0.0
    else:
        action_wrapper_score = 1.0

    if system_features["forbids_user_placeholder"]:
        anti_placeholder_score = 0.0 if placeholder_hit else 1.0
    else:
        anti_placeholder_score = 1.0 if not UNRESOLVED_USER_RE.search(raw) else 0.0

    anti_disclosure_score = 0.0 if disclosure_leak_hit else 1.0
    anti_generic_score = 0.0 if generic_assistant_hit else 1.0

    format_total = (
        0.35 * roleplay_shape
        + 0.20 * action_wrapper_score
        + 0.20 * anti_placeholder_score
        + 0.15 * anti_disclosure_score
        + 0.10 * anti_generic_score
    )

    # Blend in character-name identity signal (25 % weight).
    # Catches cross-character responses that mention the wrong character's name,
    # giving the role score a reliable identity axis beyond format markers alone.
    name_presence = score_character_name_presence(system, raw)
    total = clamp(0.75 * format_total + 0.25 * name_presence["total"])

    return {
        "total": round4(total),
        "components": {
            "roleplay_shape": round4(roleplay_shape),
            "action_wrapper_compliance": round4(action_wrapper_score),
            "anti_user_placeholder": round4(anti_placeholder_score),
            "anti_instruction_disclosure": round4(anti_disclosure_score),
            "anti_generic_assistant": round4(anti_generic_score),
            "character_name_presence": round4(name_presence["total"]),
        },
        "flags": {
            "has_user_placeholder": placeholder_hit,
            "has_disclosure_leak": disclosure_leak_hit,
            "has_generic_assistant_leak": generic_assistant_hit,
        },
        "meta": system_features,
        "name_presence": name_presence,
    }


def score_language_consistency(
    system: str,
    instruction: str,
    text: str,
    process_module,
    source_lang: Optional[str] = None,
) -> Dict[str, object]:
    raw = str(text or "")
    system_features = infer_system_features(system)
    detected_lang = process_module.detect_output_lang(raw)
    lang_eval = process_module.classify_lang_consistency(instruction, raw)

    status = str(lang_eval.get("status", "uncertain"))
    if status == "match":
        instruction_output_score = 1.0
    elif status == "uncertain":
        instruction_output_score = 0.55
    else:
        instruction_output_score = 0.0

    source_lang_score = 0.55
    if source_lang:
        if detected_lang == source_lang or process_module._same_lang_group(detected_lang, source_lang):
            source_lang_score = 1.0
        elif detected_lang == "unknown":
            source_lang_score = 0.35
        else:
            source_lang_score = 0.0

    system_rule_score = instruction_output_score if system_features["expects_language_match"] else 1.0

    total = (
        0.50 * instruction_output_score
        + 0.30 * source_lang_score
        + 0.20 * system_rule_score
    )
    total = clamp(total)

    return {
        "total": round4(total),
        "components": {
            "instruction_output_match": round4(instruction_output_score),
            "source_language_match": round4(source_lang_score),
            "system_language_rule_compliance": round4(system_rule_score),
        },
        "flags": {
            "expects_language_match": system_features["expects_language_match"],
        },
        "meta": {
            "detected_output_language": detected_lang,
            "lang_status": status,
            "lang_reason": lang_eval.get("reason"),
            "source_language": source_lang,
        },
    }


def score_length_adequacy(text: str, reference_length: Optional[int] = None) -> Dict[str, object]:
    raw = str(text or "")
    normalized = normalize_text(raw)
    char_len = len(normalized)

    if reference_length and reference_length > 0:
        ratio = char_len / max(1, reference_length)
        if 0.65 <= ratio <= 1.60:
            adequacy_score = 1.0
        elif 0.45 <= ratio <= 2.10:
            adequacy_score = 0.65
        elif 0.25 <= ratio <= 3.00:
            adequacy_score = 0.35
        else:
            adequacy_score = 0.10
    else:
        ratio = None
        if char_len >= 220:
            adequacy_score = 1.0
        elif char_len >= 120:
            adequacy_score = 0.75
        elif char_len >= 60:
            adequacy_score = 0.45
        else:
            adequacy_score = 0.15

    repetition = repeated_ngram_ratio(raw)
    anti_repetition_score = clamp(1.0 - repetition * 3.0)

    total = 0.80 * adequacy_score + 0.20 * anti_repetition_score
    total = clamp(total)

    return {
        "total": round4(total),
        "components": {
            "relative_length_fit": round4(adequacy_score),
            "anti_repetition": round4(anti_repetition_score),
        },
        "meta": {
            "char_length": char_len,
            "reference_length": reference_length,
            "length_ratio": round4(ratio) if ratio is not None else None,
            "repetition_ratio": round4(repetition),
        },
    }


_NAME_NOISE = frozenset({
    "the", "and", "but", "she", "her", "his", "him", "they", "you", "your",
    "its", "this", "that", "with", "from", "have", "has", "are", "was", "were",
    "not", "can", "will", "all", "one", "two", "for", "yes", "out",
})


def score_character_name_presence(system: str, text: str) -> Dict[str, object]:
    """Auxiliary signal for wrong_role pair validation.

    Checks whether the response references the character name found in the
    system prompt. Used at construction time only (not in online GRPO reward).

    Returns:
      total = 1.0  — own character name found in response (right character)
      total = 0.25 — a different proper noun found (wrong character name present)
      total = 0.55 — no name signal either way (neutral)
    """
    name = extract_character_name(system)
    if not name:
        return {"total": round4(0.55), "character_name": None, "found_own": False}

    raw = str(text or "")
    # Direct mention or action-prefixed: *Name verb...
    if re.search(rf"(?:^|\*|\b){re.escape(name)}\b", raw, re.IGNORECASE):
        return {"total": round4(1.0), "character_name": name, "found_own": True}

    # Another capitalised proper noun that is not the character's own name
    other_caps = [
        m.group(1) for m in re.finditer(r"\b([A-Z][a-z]{2,15})\b", raw)
        if m.group(1).lower() not in _NAME_NOISE and m.group(1) != name
    ]
    if other_caps:
        return {
            "total": round4(0.25),
            "character_name": name,
            "found_own": False,
            "other_names_found": other_caps[:4],
        }

    return {"total": round4(0.55), "character_name": name, "found_own": False}


def score_chatjoy_response(
    system: str,
    instruction: str,
    text: str,
    process_module,
    source_lang: Optional[str] = None,
    reference_length: Optional[int] = None,
    target_penalty: float = 0.0,
    format_weight: float = 0.30,
    role_weight: float = 0.30,
    language_weight: float = 0.25,
    length_weight: float = 0.15,
) -> Dict[str, object]:
    format_score = score_format_features(text)
    role_score = score_role_consistency(system=system, text=text)
    language_score = score_language_consistency(
        system=system,
        instruction=instruction,
        text=text,
        process_module=process_module,
        source_lang=source_lang,
    )
    length_score = score_length_adequacy(text=text, reference_length=reference_length)

    raw_total = (
        format_weight * format_score["total"]
        + role_weight * role_score["total"]
        + language_weight * language_score["total"]
        + length_weight * length_score["total"]
        - target_penalty
    )
    total = clamp(raw_total)
    return {
        "version": "chatjoy_format_role_language_length_v1",
        "weights": {
            "format": round4(format_weight),
            "role": round4(role_weight),
            "language": round4(language_weight),
            "length": round4(length_weight),
        },
        "target_penalty": round4(target_penalty),
        "format": format_score,
        "role_consistency": role_score,
        "language_consistency": language_score,
        "length": length_score,
        "total": round4(total),
    }
