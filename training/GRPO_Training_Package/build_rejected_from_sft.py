#!/usr/bin/env python3
"""Build preference pairs from SFT data with synthetic rejected samples.

This script uses:
1. `instruction` + `system` from clean32 SFT data as the generation context
2. the current SFT merged model to generate synthetic rejected responses
3. `process_chatjoy_data.py` language checks to validate outputs
4. ChatJoy-specific reward heuristics to score roleplay format and content quality

Supported rejected types:
- wrong_role: keep instruction, swap to a different role system prompt
- wrong_language: keep role, force a different response language
- length_mismatch: keep role/language, force too-short or too-long replies
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chatjoy_reward_utils import (
    get_scoring_profile,
    score_character_name_presence,
    score_chatjoy_response,
)


LANGUAGE_NAMES = {
    "en": "English",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "da": "Danish",
    "de": "German",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "hi": "Hindi",
    "hr": "Croatian",
    "it": "Italian",
    "ja": "Japanese",
    "kk": "Kazakh",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mr": "Marathi",
    "ms": "Malay",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sr": "Serbian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "zh-cn": "Simplified Chinese",
    "zh-tw": "Traditional Chinese",
}


def load_process_module(process_path: Path):
    spec = importlib.util.spec_from_file_location("process_chatjoy_data", process_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_records(path: Path) -> Iterator[Tuple[int, Dict]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            item = raw.get("item", raw) if isinstance(raw, dict) else None
            if isinstance(item, dict):
                yield idx, item


def count_source_roles(records: List[Tuple[int, Dict]], role_name_by_system: Dict[str, str]) -> Counter:
    counts: Counter = Counter()
    for _, item in records:
        role_name = role_name_by_system.get(str(item.get("system", "")).strip())
        if role_name:
            counts[role_name] += 1
    return counts


def count_source_languages(records: List[Tuple[int, Dict]]) -> Counter:
    counts: Counter = Counter()
    for _, item in records:
        lang = str(item.get("language", "unknown")).strip().lower()
        if lang in LANGUAGE_NAMES:
            counts[lang] += 1
    return counts


def reorder_records(
    records: List[Tuple[int, Dict]],
    role_name_by_system: Dict[str, str],
    ordering: str,
    seed: int,
) -> List[Tuple[int, Dict]]:
    if ordering == "input_order":
        return records

    if ordering != "language_role_round_robin":
        raise ValueError(f"Unsupported ordering: {ordering}")

    grouped: Dict[Tuple[str, str], List[Tuple[int, Dict]]] = {}
    for record in records:
        _, item = record
        lang = str(item.get("language", "unknown")).strip().lower()
        role = role_name_by_system.get(str(item.get("system", "")).strip(), "unknown_role")
        grouped.setdefault((lang, role), []).append(record)

    rng = random.Random(seed)
    group_items = []
    for key, group_records in grouped.items():
        shuffled_group = list(group_records)
        rng.shuffle(shuffled_group)
        group_items.append((key, shuffled_group))

    group_items.sort(key=lambda item: (-len(item[1]), item[0][0], item[0][1]))
    interleaved: List[Tuple[int, Dict]] = []
    remaining = True
    while remaining:
        remaining = False
        for _, group_records in group_items:
            if group_records:
                interleaved.append(group_records.pop())
                remaining = True
    return interleaved


def load_role_catalog(roles_dir: Path) -> List[Dict[str, str]]:
    catalog: List[Dict[str, str]] = []
    for path in sorted(roles_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        description = ""
        if isinstance(data, dict):
            description = str(data.get("description", "")).strip()
        if description:
            catalog.append({"role_id": path.stem, "system": description})
    return catalog


def resolve_input_device(model) -> str:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for device in device_map.values():
            if isinstance(device, str) and device.startswith("cuda"):
                return device
    try:
        return str(model.device)
    except Exception:
        return "cuda:0" if torch.cuda.is_available() else "cpu"


def resolve_vllm_chat_url(base_url: str) -> str:
    base = str(base_url or "").rstrip("/")
    if not base:
        raise ValueError("vLLM base URL is required when backend=vllm_openai")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def extract_message_text(message_content) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts: List[str] = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(message_content or "")


class RejectedBuilder:
    def __init__(
        self,
        model_path: Path,
        process_module,
        roles_dir: Path,
        dtype: str = "bfloat16",
        load_in_4bit: bool = True,
        seed: int = 42,
        max_input_length: int = 4096,
        max_new_tokens: int = 384,
        min_chosen_score: float = 0.62,
        min_score_gap: float = 0.18,
        backend: str = "transformers",
        tokenizer_path: Optional[Path] = None,
        vllm_base_url: Optional[str] = None,
        vllm_model: Optional[str] = None,
        vllm_api_key: str = "EMPTY",
        request_timeout: int = 180,
        request_max_retries: int = 3,
        request_retry_sleep: float = 2.0,
        length_short_prob: float = 0.5,
        wrong_role_attempts: int = 4,
        wrong_language_attempts: int = 4,
        length_attempts: int = 4,
        role_balance_mode: str = "uniform",
        language_balance_mode: str = "uniform",
        language_candidate_set: str = "observed_only",
        source_role_counts: Optional[Counter] = None,
        source_language_counts: Optional[Counter] = None,
        target_role_counts: Optional[Counter] = None,
        target_language_counts: Optional[Counter] = None,
        length_source_role_balance_mode: str = "uniform",
        length_source_language_balance_mode: str = "uniform",
        current_wrong_role_source_role_counts: Optional[Counter] = None,
        current_wrong_role_source_language_counts: Optional[Counter] = None,
        current_wrong_language_source_role_counts: Optional[Counter] = None,
        current_wrong_language_source_language_counts: Optional[Counter] = None,
        current_length_source_role_counts: Optional[Counter] = None,
        current_length_source_language_counts: Optional[Counter] = None,
        current_type_counts: Optional[Counter] = None,
        fallback_on_failure: bool = False,
        fallback_attempts: int = 2,
        fallback_min_score_gap: Optional[float] = None,
        fallback_dimension_gap_scale: float = 0.85,
        family_target_weights: Optional[Dict[str, float]] = None,
        # Zero-gen cross-reference tables (populated for cross_character / cross_lingual modes)
        all_records_by_role: Optional[Dict[str, List]] = None,
        all_records_by_lang_role: Optional[Dict[tuple, List]] = None,
    ):
        self.process = process_module
        self.roles = load_role_catalog(roles_dir)
        self.role_name_by_system = {r["system"]: r["role_id"] for r in self.roles}
        self.random = random.Random(seed)
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.min_chosen_score = min_chosen_score
        self.min_score_gap = min_score_gap
        self.backend = backend
        self.model_path = Path(model_path)
        self.request_timeout = request_timeout
        self.request_max_retries = request_max_retries
        self.request_retry_sleep = request_retry_sleep
        self.length_short_prob = length_short_prob
        self.wrong_role_attempts = max(1, wrong_role_attempts)
        self.wrong_language_attempts = max(1, wrong_language_attempts)
        self.length_attempts = max(1, length_attempts)
        self.role_balance_mode = role_balance_mode
        self.language_balance_mode = language_balance_mode
        self.language_candidate_set = language_candidate_set
        self.source_role_counts = Counter(source_role_counts or {})
        self.source_language_counts = Counter(source_language_counts or {})
        self.target_role_counts = Counter(target_role_counts or {})
        self.target_language_counts = Counter(target_language_counts or {})
        self.length_source_role_balance_mode = length_source_role_balance_mode
        self.length_source_language_balance_mode = length_source_language_balance_mode
        self.current_wrong_role_source_role_counts = Counter(current_wrong_role_source_role_counts or {})
        self.current_wrong_role_source_language_counts = Counter(current_wrong_role_source_language_counts or {})
        self.current_wrong_language_source_role_counts = Counter(current_wrong_language_source_role_counts or {})
        self.current_wrong_language_source_language_counts = Counter(current_wrong_language_source_language_counts or {})
        self.current_length_source_role_counts = Counter(current_length_source_role_counts or {})
        self.current_length_source_language_counts = Counter(current_length_source_language_counts or {})
        self.current_type_counts = Counter(current_type_counts or {})
        self.fallback_on_failure = fallback_on_failure
        self.fallback_attempts = max(1, fallback_attempts)
        self.fallback_min_score_gap = fallback_min_score_gap
        self.fallback_dimension_gap_scale = fallback_dimension_gap_scale
        self.family_target_weights = dict(
            family_target_weights
            or {
                "wrong_role": 0.50,
                "wrong_language": 0.40,
                "length_mismatch": 0.10,
            }
        )
        self.vllm_base_url = str(vllm_base_url or "").rstrip("/") or None
        self.vllm_api_key = vllm_api_key
        self.vllm_model = vllm_model or str(model_path)
        self.vllm_chat_url = resolve_vllm_chat_url(self.vllm_base_url) if self.backend == "vllm_openai" else None
        self.tokenizer = None
        self.model = None
        self.input_device = None
        # Zero-gen cross-reference tables
        self.all_records_by_role: Dict[str, List] = dict(all_records_by_role or {})
        self.all_records_by_lang_role: Dict[tuple, List] = dict(all_records_by_lang_role or {})

        if self.backend == "transformers":
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map[dtype]

            tokenizer_source = str(tokenizer_path or model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
                "torch_dtype": torch_dtype,
            }
            if load_in_4bit:
                model_kwargs["load_in_4bit"] = True

            self.model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
            self.model.eval()
            self.input_device = resolve_input_device(self.model)
        elif self.backend == "vllm_openai":
            if not self.vllm_base_url:
                raise ValueError("--vllm-base-url is required when --backend vllm_openai")
        elif self.backend == "none":
            pass  # zero-gen modes: no model needed
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def choose_type(self, wrong_role_weight: float, wrong_language_weight: float, length_weight: float) -> str:
        total = wrong_role_weight + wrong_language_weight + length_weight
        p = self.random.random() * total
        if p < wrong_role_weight:
            return "wrong_role"
        if p < wrong_role_weight + wrong_language_weight:
            return "wrong_language"
        return "length_mismatch"

    @staticmethod
    def type_family(rejected_type: str) -> str:
        if rejected_type.startswith("length_mismatch"):
            return "length_mismatch"
        return rejected_type

    def attempts_for_type(self, rejected_type: str, fallback: bool = False) -> int:
        if fallback:
            return self.fallback_attempts
        family = self.type_family(rejected_type)
        if family == "wrong_role":
            return self.wrong_role_attempts
        if family == "wrong_language":
            return self.wrong_language_attempts
        return self.length_attempts

    def current_family_counts(self) -> Counter:
        family_counts = Counter()
        for key, value in self.current_type_counts.items():
            family_counts[self.type_family(key)] += value
        return family_counts

    def family_deficit_order(self, exclude: Optional[set[str]] = None) -> List[str]:
        exclude = exclude or set()
        family_counts = self.current_family_counts()
        total = sum(family_counts.values())
        candidates = []
        for family, target_weight in self.family_target_weights.items():
            if family in exclude:
                continue
            current_share = (family_counts.get(family, 0) / total) if total > 0 else 0.0
            deficit = target_weight - current_share
            candidates.append((deficit, self.random.random(), family))
        candidates.sort(reverse=True)
        return [family for _, _, family in candidates]

    def family_share(self, family: str) -> float:
        family_counts = self.current_family_counts()
        total = sum(family_counts.values())
        if total <= 0:
            return 0.0
        return family_counts.get(family, 0) / total

    def family_weight_multiplier(self, family: str) -> float:
        target = self.family_target_weights.get(family, 0.0)
        current = self.family_share(family)
        delta = current - target

        if family == "length_mismatch":
            if delta >= 0.12:
                return 0.02
            if delta >= 0.08:
                return 0.05
            if delta >= 0.04:
                return 0.15
            return 1.0

        if delta <= -0.10:
            return 1.9
        if delta <= -0.06:
            return 1.6
        if delta <= -0.03:
            return 1.3
        if delta >= 0.06:
            return 0.55
        return 1.0

    def fallback_sequence(self, primary_type: str) -> List[str]:
        primary_family = self.type_family(primary_type)
        if primary_family == "wrong_role":
            return ["wrong_language", "length_mismatch"]
        if primary_family == "wrong_language":
            return ["wrong_role", "length_mismatch"]
        ordered = self.family_deficit_order(exclude={primary_family})
        return ordered

    def generate(self, system: str, instruction: str, temperature: float = 0.9, top_p: float = 0.95) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]

        if self.backend == "vllm_openai":
            payload = {
                "model": self.vllm_model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": self.max_new_tokens,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.vllm_api_key}",
            }
            last_error: Optional[Exception] = None
            for attempt in range(1, self.request_max_retries + 1):
                try:
                    request = urllib.request.Request(
                        self.vllm_chat_url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers=headers,
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                        data = json.loads(response.read().decode("utf-8"))
                    choices = data.get("choices") or []
                    if not choices:
                        raise RuntimeError(f"vLLM returned no choices: {data}")
                    content = extract_message_text(choices[0].get("message", {}).get("content", ""))
                    return content.strip()
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="ignore")
                    last_error = RuntimeError(f"HTTP {exc.code}: {body}")
                except Exception as exc:  # noqa: BLE001
                    last_error = exc

                if attempt < self.request_max_retries:
                    time.sleep(self.request_retry_sleep)

            raise RuntimeError(f"vLLM generation failed after {self.request_max_retries} attempts: {last_error}")

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        inputs = {k: v.to(self.input_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def choose_balanced_target(
        self,
        candidates: List[str],
        current_counts: Counter,
        source_counts: Counter,
        mode: str,
    ) -> str:
        if not candidates:
            raise ValueError("No candidates available")

        if mode == "source_proportional":
            weighted_candidates = []
            for candidate in candidates:
                quota = max(1.0, float(source_counts.get(candidate, 0)))
                score = current_counts.get(candidate, 0) / quota
                weighted_candidates.append((score, candidate))
        else:
            weighted_candidates = [
                (float(current_counts.get(candidate, 0)), candidate)
                for candidate in candidates
            ]

        best_score = min(score for score, _ in weighted_candidates)
        best_candidates = [candidate for score, candidate in weighted_candidates if score == best_score]
        return self.random.choice(best_candidates)

    def candidate_balance_score(
        self,
        candidate: str,
        current_counts: Counter,
        source_counts: Counter,
        mode: str,
    ) -> float:
        if mode == "source_proportional":
            quota = max(1.0, float(source_counts.get(candidate, 0)))
            return current_counts.get(candidate, 0) / quota
        return float(current_counts.get(candidate, 0))

    def source_is_overrepresented(
        self,
        candidate: str,
        current_counts: Counter,
        source_counts: Counter,
        mode: str,
    ) -> bool:
        candidates = [key for key, count in source_counts.items() if count > 0]
        if candidate not in source_counts or not candidates:
            return False
        candidate_score = self.candidate_balance_score(candidate, current_counts, source_counts, mode)
        all_scores = [self.candidate_balance_score(key, current_counts, source_counts, mode) for key in candidates]
        min_score = min(all_scores)
        slack = 0.35 if mode == "source_proportional" else 1.0
        return candidate_score > (min_score + slack)

    def should_avoid_length_source(self, chosen_role: str, source_lang: str) -> bool:
        role_over = False
        lang_over = False
        if chosen_role and chosen_role in self.source_role_counts:
            role_over = self.source_is_overrepresented(
                chosen_role,
                current_counts=self.current_length_source_role_counts,
                source_counts=self.source_role_counts,
                mode=self.length_source_role_balance_mode,
            )
        if source_lang and source_lang in self.source_language_counts:
            lang_over = self.source_is_overrepresented(
                source_lang,
                current_counts=self.current_length_source_language_counts,
                source_counts=self.source_language_counts,
                mode=self.length_source_language_balance_mode,
            )
        return role_over or lang_over

    def should_avoid_wrong_role_source(self, chosen_role: str, source_lang: str) -> bool:
        role_over = False
        lang_over = False
        if chosen_role and chosen_role in self.source_role_counts:
            role_over = self.source_is_overrepresented(
                chosen_role,
                current_counts=self.current_wrong_role_source_role_counts,
                source_counts=self.source_role_counts,
                mode="source_proportional",
            )
        if source_lang and source_lang in self.source_language_counts:
            lang_over = self.source_is_overrepresented(
                source_lang,
                current_counts=self.current_wrong_role_source_language_counts,
                source_counts=self.source_language_counts,
                mode="source_proportional",
            )
        return role_over or lang_over

    def should_avoid_wrong_language_source(self, chosen_role: str, source_lang: str) -> bool:
        role_over = False
        lang_over = False
        if chosen_role and chosen_role in self.source_role_counts:
            role_over = self.source_is_overrepresented(
                chosen_role,
                current_counts=self.current_wrong_language_source_role_counts,
                source_counts=self.source_role_counts,
                mode="source_proportional",
            )
        if source_lang and source_lang in self.source_language_counts:
            lang_over = self.source_is_overrepresented(
                source_lang,
                current_counts=self.current_wrong_language_source_language_counts,
                source_counts=self.source_language_counts,
                mode="source_proportional",
            )
        return role_over or lang_over

    def choose_type_for_item(
        self,
        item: Dict,
        wrong_role_weight: float,
        wrong_language_weight: float,
        length_weight: float,
        ) -> str:
        chosen_role = self.role_name_by_system.get(str(item.get("system", "")).strip(), "unknown_role")
        source_lang = str(item.get("language", "unknown")).strip().lower()
        adjusted_wrong_role = wrong_role_weight * self.family_weight_multiplier("wrong_role")
        adjusted_wrong_role *= 0.35 if self.should_avoid_wrong_role_source(chosen_role, source_lang) else 1.0
        adjusted_wrong_language = wrong_language_weight * self.family_weight_multiplier("wrong_language")
        adjusted_wrong_language *= 0.35 if self.should_avoid_wrong_language_source(chosen_role, source_lang) else 1.0
        adjusted_length = length_weight * self.family_weight_multiplier("length_mismatch")
        adjusted_length *= 0.20 if self.should_avoid_length_source(chosen_role, source_lang) else 1.0

        if adjusted_wrong_role + adjusted_wrong_language + adjusted_length <= 0:
            return self.choose_type(wrong_role_weight, wrong_language_weight, length_weight)
        return self.choose_type(adjusted_wrong_role, adjusted_wrong_language, adjusted_length)

    def pick_wrong_role_system(self, current_system: str) -> Tuple[str, str]:
        choices = [r for r in self.roles if r["system"] != current_system]
        if not choices:
            raise ValueError("No alternate role systems available")
        candidate_ids = [choice["role_id"] for choice in choices]
        chosen_role_id = self.choose_balanced_target(
            candidate_ids,
            current_counts=self.target_role_counts,
            source_counts=self.source_role_counts,
            mode=self.role_balance_mode,
        )
        chosen = next(choice for choice in choices if choice["role_id"] == chosen_role_id)
        return chosen["system"], chosen["role_id"]

    def pick_wrong_language(self, source_lang: str) -> Tuple[str, str]:
        if self.language_candidate_set == "observed_only":
            candidates = [code for code in self.source_language_counts if code != source_lang and code in LANGUAGE_NAMES]
        else:
            candidates = [code for code in LANGUAGE_NAMES if code != source_lang]
        if not candidates:
            candidates = [code for code in LANGUAGE_NAMES if code != source_lang]
        lang_code = self.choose_balanced_target(
            candidates,
            current_counts=self.target_language_counts,
            source_counts=self.source_language_counts,
            mode=self.language_balance_mode,
        )
        return lang_code, LANGUAGE_NAMES[lang_code]

    @staticmethod
    def reference_reply_block(chosen_text: str, max_chars: int = 1400) -> str:
        text = (chosen_text or "").strip()
        if not text:
            return ""
        if len(text) > max_chars:
            text = text[:max_chars].rstrip()
        return "\n\nReference reply to transform (preserve scene facts and formatting rhythm, but do not copy verbatim):\n" + text

    def build_wrong_language_system(self, original_system: str, target_lang_name: str, source_lang_name: str) -> str:
        # Append a language-override directive to the real character system prompt.
        # The model responds naturally as the correct character, with only the
        # output language forced to be wrong.  No reference reply needed.
        override = (
            f"\n\n[LANGUAGE OVERRIDE: You must respond ONLY in {target_lang_name}. "
            f"Do not use {source_lang_name} at all. "
            "Keep your character identity, scene, and roleplay format exactly as specified above. "
            "Only the output language must change.]"
        )
        return original_system + override

    def extract_role_style(self, system_text: str, max_chars: int = 1200) -> str:
        lines = system_text.splitlines()
        captured: List[str] = []
        in_personality = False
        stop_headers = (
            "# FIRST MESSAGE",
            "# INITIAL MESSAGES",
            "# EXAMPLE",
            "# SCENARIO",
            "# STARTING SCENARIO",
            "# SETTING",
        )

        for raw_line in lines:
            line = raw_line.rstrip()
            upper = line.strip().upper()
            if upper.startswith("# PERSONALITY"):
                in_personality = True
                continue
            if in_personality and any(upper.startswith(header) for header in stop_headers):
                break
            if in_personality and line.strip():
                captured.append(line)

        text = "\n".join(captured) if captured else system_text
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) > max_chars:
            text = text[:max_chars].rsplit("\n", 1)[0].strip()
        return text

    def extract_shared_system_scaffold(self, system_text: str, max_chars: int = 1800) -> str:
        text = str(system_text or "").strip()
        if not text:
            return ""

        if "\n---" in text:
            text = text.split("\n---", 1)[0].strip()
        elif "\n# Sample Messages" in text:
            text = text.split("\n# Sample Messages", 1)[0].strip()

        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) > max_chars:
            text = text[:max_chars].rsplit("\n", 1)[0].strip()
        return text

    def build_wrong_role_system(self, original_system: str, alt_system: str, alt_role_name: str, chosen_text: str) -> str:
        shared_scaffold = self.extract_shared_system_scaffold(original_system)
        alt_style = self.extract_role_style(alt_system)
        prefix = (
            "Synthetic negative sample instruction: keep the user's current task, scene facts, immediate situation, "
            "language, and roleplay formatting habits consistent with the source conversation. Do NOT import any new "
            "names, locations, relationships, backstory, first-message setup, or worldbuilding from another card. "
            "Use the shared constraints and scene scaffold below, but do not follow the source character identity or "
            "source character biography. Rewrite close to the reference reply's structure and semantic content, while "
            "changing only the assistant's persona, attitude, social dynamic, and speaking style so the reply clearly "
            f"feels like the wrong character. The wrong role style target is: {alt_role_name}. "
            "Make the style change obvious by shifting terms of address, emotional temperature, flirtation or distance, "
            "dominance or softness, sentence rhythm, punctuation habits, and habitual phrasing toward the target role. "
            "If the reply could still plausibly pass as the source character, that is a failure."
        )
        shared_block = "Shared constraints and scene scaffold from source card:\n" + shared_scaffold
        style_block = "Wrong-role style target (style only, not scenario/worldbuilding):\n" + alt_style
        return prefix + "\n\n" + shared_block + "\n\n" + style_block + self.reference_reply_block(chosen_text)

    def build_length_system(self, original_system: str, chosen_len: int, mode: str) -> Tuple[str, float]:
        if mode == "short":
            target_chars = max(24, int(chosen_len * self.random.uniform(0.25, 0.5)))
            prefix = (
                "Synthetic negative sample instruction: respond in the same role and same language, "
                f"but make the answer unnaturally brief and incomplete. Target around {target_chars} characters."
            )
            return prefix + "\n\n" + original_system, 0.40

        target_chars = max(chosen_len + 80, int(chosen_len * self.random.uniform(1.5, 2.0)))
        prefix = (
            "Synthetic negative sample instruction: respond in the same role and same language, "
            f"but make the answer overly verbose, repetitive, and longer than necessary. Target around {target_chars} characters."
        )
        return prefix + "\n\n" + original_system, 0.55

    def validate_wrong_role(self, instruction: str, rejected: str, source_lang: str) -> bool:
        if not rejected or self.process.output_has_user_placeholder(rejected):
            return False
        lang_eval = self.process.classify_lang_consistency(instruction, rejected)
        detected = self.process.detect_output_lang(rejected)
        return lang_eval.get("status") == "match" and (
            detected == source_lang or self.process._same_lang_group(detected, source_lang)
        )

    def validate_wrong_language(self, instruction: str, rejected: str, source_lang: str) -> bool:
        if not rejected or self.process.output_has_user_placeholder(rejected):
            return False
        detected = self.process.detect_output_lang(rejected)
        if detected == "unknown":
            return False
        if detected == source_lang or self.process._same_lang_group(detected, source_lang):
            return False
        return True

    def validate_length(self, instruction: str, rejected: str, source_lang: str, chosen_len: int, mode: str) -> bool:
        if not rejected or self.process.output_has_user_placeholder(rejected):
            return False
        lang_eval = self.process.classify_lang_consistency(instruction, rejected)
        if lang_eval.get("status") != "match":
            return False
        length = len(rejected)
        if mode == "short":
            return length < max(40, int(chosen_len * 0.7))
        return length > int(chosen_len * 1.2)

    def target_penalty(self, rejected_type: str) -> float:
        if rejected_type == "wrong_language":
            return 0.34
        if rejected_type == "wrong_role":
            return 0.30
        if rejected_type == "length_mismatch_short":
            return 0.18
        if rejected_type == "length_mismatch_long":
            return 0.12
        return 0.15

    def score_output(
        self,
        system: str,
        instruction: str,
        text: str,
        source_lang: str,
        reference_length: int,
        penalty: float = 0.0,
        rejected_type: Optional[str] = None,
    ):
        profile = get_scoring_profile(rejected_type) if rejected_type else {}
        return score_chatjoy_response(
            system=system,
            instruction=instruction,
            text=text,
            process_module=self.process,
            source_lang=source_lang,
            reference_length=reference_length,
            target_penalty=penalty,
            **profile,
        )

    def targeted_dimension_gap(self, chosen_reward: Dict, rejected_reward: Dict, rejected_type: str) -> Tuple[str, float]:
        if rejected_type == "wrong_role":
            key = "role_consistency"
        elif rejected_type == "wrong_language":
            key = "language_consistency"
        else:
            key = "length"
        gap = chosen_reward[key]["total"] - rejected_reward[key]["total"]
        return key, round(gap, 4)

    def min_dimension_gap(self, rejected_type: str) -> float:
        if rejected_type == "wrong_language":
            return 0.30
        if rejected_type == "wrong_role":
            return 0.12  # raised from 0.08: name signal makes valid pairs reliable
        if rejected_type.startswith("length_mismatch"):
            return 0.20
        return 0.18

    def effective_min_score_gap(self, rejected_type: str, fallback: bool = False) -> float:
        if rejected_type == "wrong_role":
            if fallback:
                return 0.13
            return 0.14
        if fallback and self.fallback_min_score_gap is not None:
            return self.fallback_min_score_gap
        return self.min_score_gap

    def effective_min_dimension_gap(self, rejected_type: str, fallback: bool = False) -> float:
        base_gap = self.min_dimension_gap(rejected_type)
        if fallback:
            return max(0.12, round(base_gap * self.fallback_dimension_gap_scale, 4))
        return base_gap

    def try_build_pair_type(
        self,
        index: int,
        item: Dict,
        rejected_type: str,
        max_attempts: int,
        *,
        primary_type: str,
        fallback: bool = False,
    ) -> Optional[Dict]:
        instruction = str(item.get("instruction", "")).strip()
        chosen = str(item.get("output", "")).strip()
        system = str(item.get("system", "")).strip()
        source_lang = str(item.get("language", "unknown")).strip().lower()
        if not instruction or not chosen or not system:
            return None

        chosen_role = self.role_name_by_system.get(system, "unknown_role")
        chosen_len = len(chosen)
        chosen_reward = self.score_output(system, instruction, chosen, source_lang, chosen_len, penalty=0.0, rejected_type=rejected_type)
        if chosen_reward["total"] < self.min_chosen_score:
            return None

        for _ in range(max_attempts):
            meta: Dict[str, object] = {
                "source_index": index,
                "language": source_lang,
                "chosen_role": chosen_role,
                "primary_rejected_type": primary_type,
                "generation_phase": "fallback" if fallback else "primary",
            }
            final_rejected_type = rejected_type

            if rejected_type == "wrong_role":
                alt_system, rejected_role = self.pick_wrong_role_system(system)
                # Generate with the other character's real system prompt so the model
                # naturally responds as character B to prompt A.  Scoring is still done
                # under the original system (system_A), so role/name signals fire cleanly.
                rejected = self.generate(alt_system, instruction, temperature=0.85, top_p=0.92)
                if not self.validate_wrong_role(instruction, rejected, source_lang):
                    continue
                meta.update(
                    {
                        "rejected_role": rejected_role,
                        "detected_rejected_language": self.process.detect_output_lang(rejected),
                        "alt_system_excerpt": alt_system[:200],
                    }
                )
                reason = f"wrong_role: {chosen_role} -> {rejected_role}"
                rejected_system = alt_system

            elif rejected_type == "wrong_language":
                target_code, target_name = self.pick_wrong_language(source_lang)
                source_lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
                wrong_lang_system = self.build_wrong_language_system(system, target_name, source_lang_name)
                rejected = self.generate(wrong_lang_system, instruction, temperature=0.85, top_p=0.92)
                if not self.validate_wrong_language(instruction, rejected, source_lang):
                    continue
                meta.update(
                    {
                        "target_wrong_language": target_code,
                        "detected_rejected_language": self.process.detect_output_lang(rejected),
                    }
                )
                reason = f"wrong_language: {source_lang} -> {meta['detected_rejected_language']}"
                rejected_system = wrong_lang_system

            else:
                mode = "short" if self.random.random() < self.length_short_prob else "long"
                length_system, _ = self.build_length_system(system, chosen_len, mode)
                rejected = self.generate(length_system, instruction, temperature=0.92, top_p=0.95)
                if not self.validate_length(instruction, rejected, source_lang, chosen_len, mode):
                    continue
                meta.update(
                    {
                        "length_mode": mode,
                        "chosen_length": chosen_len,
                        "rejected_length": len(rejected),
                        "detected_rejected_language": self.process.detect_output_lang(rejected),
                    }
                )
                reason = f"length_mismatch: {mode}"
                rejected_system = length_system
                final_rejected_type = f"length_mismatch_{mode}"

            rejected_reward = self.score_output(
                system,
                instruction,
                rejected,
                source_lang,
                chosen_len,
                penalty=self.target_penalty(final_rejected_type),
                rejected_type=final_rejected_type,
            )
            score_gap = round(chosen_reward["total"] - rejected_reward["total"], 4)
            if score_gap < self.effective_min_score_gap(final_rejected_type, fallback=fallback):
                continue

            dimension_key, dimension_gap = self.targeted_dimension_gap(
                chosen_reward,
                rejected_reward,
                final_rejected_type,
            )
            if dimension_gap < self.effective_min_dimension_gap(final_rejected_type, fallback=fallback):
                continue

            # For wrong_role: when a character name is extractable, require that
            # the name signal is in the right direction — chosen mentions its own
            # name (found_own=True) and rejected does not (found_own=False).
            # If this stricter condition fails, require a larger role gap (≥ 0.15)
            # to ensure the pair carries a meaningful identity signal.
            name_check_chosen = None
            name_check_rejected = None
            if final_rejected_type == "wrong_role":
                name_check_chosen = score_character_name_presence(system, chosen)
                name_check_rejected = score_character_name_presence(system, rejected)
                name_extractable = name_check_chosen.get("character_name") is not None
                if name_extractable:
                    good_direction = (
                        name_check_chosen.get("found_own", False)
                        and not name_check_rejected.get("found_own", False)
                    )
                    if not good_direction and dimension_gap < 0.15:
                        continue

            meta.update(
                {
                    "reward_design_version": chosen_reward["version"],
                    "reward_gap": score_gap,
                    "target_dimension": dimension_key,
                    "target_dimension_gap": dimension_gap,
                    "chosen_reward": chosen_reward,
                    "rejected_reward": rejected_reward,
                }
            )
            if name_check_chosen is not None:
                meta["name_check_chosen"] = name_check_chosen
                meta["name_check_rejected"] = name_check_rejected

            if final_rejected_type == "wrong_role":
                self.target_role_counts[meta["rejected_role"]] += 1
                self.current_wrong_role_source_role_counts[chosen_role] += 1
                self.current_wrong_role_source_language_counts[source_lang] += 1
            elif final_rejected_type == "wrong_language":
                self.target_language_counts[meta["target_wrong_language"]] += 1
                self.current_wrong_language_source_role_counts[chosen_role] += 1
                self.current_wrong_language_source_language_counts[source_lang] += 1
            elif final_rejected_type.startswith("length_mismatch"):
                self.current_length_source_role_counts[chosen_role] += 1
                self.current_length_source_language_counts[source_lang] += 1

            self.current_type_counts[final_rejected_type] += 1

            return {
                "prompt": instruction,
                "system": system,
                "chosen": chosen,
                "rejected": rejected,
                "language": source_lang,
                "rejected_type": final_rejected_type,
                "quality_chosen": chosen_reward["total"],
                "quality_rejected": rejected_reward["total"],
                "reason": reason,
                "metadata": meta,
                "chosen_system": system,
                "rejected_system": rejected_system,
                "source": "sft_rejected_builder",
            }

        return None

    def build_pair(self, index: int, item: Dict, rejected_type: str, max_attempts: int = 4) -> Optional[Dict]:
        sequence: List[Tuple[str, bool, int]] = [
            (rejected_type, False, self.attempts_for_type(rejected_type, fallback=False))
        ]
        if self.fallback_on_failure:
            for fallback_type in self.fallback_sequence(rejected_type):
                sequence.append((fallback_type, True, self.attempts_for_type(fallback_type, fallback=True)))

        seen: set[str] = set()
        for candidate_type, is_fallback, attempts in sequence:
            family = self.type_family(candidate_type)
            if family in seen:
                continue
            seen.add(family)
            pair = self.try_build_pair_type(
                index,
                item,
                candidate_type,
                max_attempts=max(1, min(max_attempts, attempts)) if not is_fallback else attempts,
                primary_type=rejected_type,
                fallback=is_fallback,
            )
            if pair is not None:
                return pair
        return None

    # ------------------------------------------------------------------
    # Zero-generation builders (no model inference required)
    # ------------------------------------------------------------------

    def build_cross_character_pair(self, index: int, item: Dict) -> Optional[Dict]:
        """Wrong-role pair via cross-character SFT data swap (zero-generation).

        Picks a real SFT response from a *different* character and scores it
        against the original character's system prompt.  Because the text was
        written for the wrong character it naturally exposes role-identity
        signals that the purely format-based heuristic can't manufacture.
        """
        instruction = str(item.get("instruction", "")).strip()
        chosen = str(item.get("output", "")).strip()
        system = str(item.get("system", "")).strip()
        source_lang = str(item.get("language", "unknown")).strip().lower()
        if not instruction or not chosen or not system:
            return None

        chosen_role = self.role_name_by_system.get(system, "unknown_role")
        chosen_len = len(chosen)

        chosen_reward = self.score_output(
            system, instruction, chosen, source_lang, chosen_len,
            penalty=0.0, rejected_type="wrong_role",
        )
        if chosen_reward["total"] < self.min_chosen_score:
            return None

        # Collect candidates: other characters, same language
        candidates: List[Tuple[str, Dict]] = []
        for role_id, role_records in self.all_records_by_role.items():
            if role_id == chosen_role:
                continue
            for record in role_records:
                if str(record.get("language", "")).lower() == source_lang:
                    candidates.append((role_id, record))

        self.random.shuffle(candidates)

        for rejected_role_id, candidate in candidates[:30]:
            rejected_text = str(candidate.get("output", "")).strip()
            if not rejected_text or rejected_text == chosen:
                continue
            if self.process.output_has_user_placeholder(rejected_text):
                continue

            rejected_reward = self.score_output(
                system, instruction, rejected_text, source_lang, chosen_len,
                penalty=0.30, rejected_type="wrong_role",
            )
            score_gap = round(chosen_reward["total"] - rejected_reward["total"], 4)
            if score_gap < 0.12:
                continue

            role_gap = round(
                chosen_reward["role_consistency"]["total"] - rejected_reward["role_consistency"]["total"], 4
            )
            if role_gap < 0.06:
                continue

            name_check = score_character_name_presence(system, rejected_text)

            self.target_role_counts[rejected_role_id] += 1
            self.current_wrong_role_source_role_counts[chosen_role] += 1
            self.current_wrong_role_source_language_counts[source_lang] += 1
            self.current_type_counts["wrong_role"] += 1

            return {
                "prompt": instruction,
                "system": system,
                "chosen": chosen,
                "rejected": rejected_text,
                "language": source_lang,
                "rejected_type": "wrong_role",
                "quality_chosen": chosen_reward["total"],
                "quality_rejected": rejected_reward["total"],
                "reason": f"cross_character: {chosen_role} -> {rejected_role_id}",
                "metadata": {
                    "source_index": index,
                    "language": source_lang,
                    "chosen_role": chosen_role,
                    "rejected_role": rejected_role_id,
                    "primary_rejected_type": "wrong_role",
                    "generation_phase": "zero_gen_cross_character",
                    "reward_gap": score_gap,
                    "target_dimension": "role_consistency",
                    "target_dimension_gap": role_gap,
                    "chosen_reward": chosen_reward,
                    "rejected_reward": rejected_reward,
                    "character_name_check": name_check,
                    "reward_design_version": chosen_reward.get("version", ""),
                },
                "chosen_system": system,
                "rejected_system": str(candidate.get("system", "")),
                "source": "cross_character_zero_gen",
            }

        return None

    def build_cross_lingual_pair(self, index: int, item: Dict) -> Optional[Dict]:
        """Wrong-language pair via cross-lingual SFT data swap (zero-generation).

        Picks a real SFT response from the *same* character but in a different
        language and scores it against the source instruction.  Language
        mismatch is guaranteed by the data label, not by detection alone.
        """
        instruction = str(item.get("instruction", "")).strip()
        chosen = str(item.get("output", "")).strip()
        system = str(item.get("system", "")).strip()
        source_lang = str(item.get("language", "unknown")).strip().lower()
        if not instruction or not chosen or not system:
            return None

        chosen_role = self.role_name_by_system.get(system, "unknown_role")
        chosen_len = len(chosen)

        chosen_reward = self.score_output(
            system, instruction, chosen, source_lang, chosen_len,
            penalty=0.0, rejected_type="wrong_language",
        )
        if chosen_reward["total"] < self.min_chosen_score:
            return None

        # Candidates: same character, different language
        candidates: List[Tuple[str, Dict]] = []
        for (lang, role_id), records in self.all_records_by_lang_role.items():
            if role_id == chosen_role and lang != source_lang and lang not in ("unknown",):
                for r in records:
                    candidates.append((lang, r))

        self.random.shuffle(candidates)

        for candidate_lang, candidate in candidates[:30]:
            rejected_text = str(candidate.get("output", "")).strip()
            if not rejected_text or rejected_text == chosen:
                continue
            if self.process.output_has_user_placeholder(rejected_text):
                continue

            # Confirm the output language is genuinely different from source
            detected = self.process.detect_output_lang(rejected_text)
            if detected == "unknown":
                continue
            if detected == source_lang or self.process._same_lang_group(detected, source_lang):
                continue

            rejected_reward = self.score_output(
                system, instruction, rejected_text, source_lang, chosen_len,
                penalty=0.34, rejected_type="wrong_language",
            )
            score_gap = round(chosen_reward["total"] - rejected_reward["total"], 4)
            if score_gap < 0.12:
                continue

            lang_gap = round(
                chosen_reward["language_consistency"]["total"] - rejected_reward["language_consistency"]["total"], 4
            )
            if lang_gap < 0.20:
                continue

            self.target_language_counts[candidate_lang] += 1
            self.current_wrong_language_source_role_counts[chosen_role] += 1
            self.current_wrong_language_source_language_counts[source_lang] += 1
            self.current_type_counts["wrong_language"] += 1

            return {
                "prompt": instruction,
                "system": system,
                "chosen": chosen,
                "rejected": rejected_text,
                "language": source_lang,
                "rejected_type": "wrong_language",
                "quality_chosen": chosen_reward["total"],
                "quality_rejected": rejected_reward["total"],
                "reason": f"cross_lingual: {source_lang} -> {detected}",
                "metadata": {
                    "source_index": index,
                    "language": source_lang,
                    "chosen_role": chosen_role,
                    "primary_rejected_type": "wrong_language",
                    "generation_phase": "zero_gen_cross_lingual",
                    "target_wrong_language": candidate_lang,
                    "detected_rejected_language": detected,
                    "reward_gap": score_gap,
                    "target_dimension": "language_consistency",
                    "target_dimension_gap": lang_gap,
                    "chosen_reward": chosen_reward,
                    "rejected_reward": rejected_reward,
                    "reward_design_version": chosen_reward.get("version", ""),
                },
                "chosen_system": system,
                "rejected_system": system,
                "source": "cross_lingual_zero_gen",
            }

        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rejected samples from SFT data")
    # model-path is optional when --backend none (zero-gen modes)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    # For zero-gen modes, load cross-reference data from a separate file if needed
    # (defaults to --input when absent)
    parser.add_argument("--all-records-path", type=str, default=None)
    parser.add_argument(
        "--roles-dir",
        type=str,
        default="/home/ps/hanjiaqi/LlamaFactory/chatjoy/TOP20CHARACTERS_systemprompt",
    )
    parser.add_argument(
        "--process-script",
        type=str,
        default="/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/process_chatjoy_data.py",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="mixed",
        choices=[
            "mixed",
            "wrong_role",
            "wrong_language",
            "length_mismatch",
            "cross_character",   # zero-gen wrong_role
            "cross_lingual",     # zero-gen wrong_language
        ],
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input-ordering",
        type=str,
        default="input_order",
        choices=["input_order", "language_role_round_robin"],
    )
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-input-length", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm_openai", "none"],
        # "none" skips model loading entirely; required for cross_character / cross_lingual modes
    )
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--vllm-base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--vllm-model", type=str, default=None)
    parser.add_argument("--vllm-api-key", type=str, default="EMPTY")
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--request-max-retries", type=int, default=3)
    parser.add_argument("--request-retry-sleep", type=float, default=2.0)
    parser.add_argument("--length-short-prob", type=float, default=0.5)
    parser.add_argument("--wrong-role-attempts", type=int, default=4)
    parser.add_argument("--wrong-language-attempts", type=int, default=4)
    parser.add_argument("--length-attempts", type=int, default=4)
    parser.add_argument("--fallback-on-failure", action="store_true")
    parser.add_argument("--fallback-attempts", type=int, default=2)
    parser.add_argument("--fallback-min-score-gap", type=float, default=None)
    parser.add_argument("--fallback-dimension-gap-scale", type=float, default=0.85)
    parser.add_argument(
        "--wrong-role-balance-mode",
        type=str,
        default="uniform",
        choices=["uniform", "source_proportional"],
    )
    parser.add_argument(
        "--wrong-language-balance-mode",
        type=str,
        default="uniform",
        choices=["uniform", "source_proportional"],
    )
    parser.add_argument(
        "--wrong-language-candidate-set",
        type=str,
        default="observed_only",
        choices=["observed_only", "all_known"],
    )
    parser.add_argument(
        "--length-source-role-balance-mode",
        type=str,
        default="uniform",
        choices=["uniform", "source_proportional"],
    )
    parser.add_argument(
        "--length-source-language-balance-mode",
        type=str,
        default="uniform",
        choices=["uniform", "source_proportional"],
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--skip-existing-output", action="store_true")
    parser.add_argument("--wrong-role-weight", type=float, default=0.45)
    parser.add_argument("--wrong-language-weight", type=float, default=0.20)
    parser.add_argument("--length-weight", type=float, default=0.35)
    parser.add_argument("--min-chosen-score", type=float, default=0.55)
    parser.add_argument("--min-score-gap", type=float, default=0.18)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.set_defaults(load_in_4bit=True)
    parser.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true")
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    return parser.parse_args()


def checkpoint_payload(
    *,
    args: argparse.Namespace,
    output_path: Path,
    report_path: Path,
    checkpoint_path: Path,
    total_records: int,
    assigned_records: int,
    start_pos: int,
    next_pos: int,
    counters: Counter,
    started_at: float,
    completed: bool,
) -> Dict:
    return {
        "input": args.input,
        "output": str(output_path),
        "report": str(report_path),
        "checkpoint": str(checkpoint_path),
        "backend": args.backend,
        "mode": args.mode,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "input_ordering": args.input_ordering,
        "max_attempts": args.max_attempts,
        "wrong_role_attempts": args.wrong_role_attempts,
        "wrong_language_attempts": args.wrong_language_attempts,
        "length_attempts": args.length_attempts,
        "min_chosen_score": args.min_chosen_score,
        "min_score_gap": args.min_score_gap,
        "fallback_on_failure": args.fallback_on_failure,
        "fallback_attempts": args.fallback_attempts,
        "fallback_min_score_gap": args.fallback_min_score_gap,
        "fallback_dimension_gap_scale": args.fallback_dimension_gap_scale,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "input_records_total": total_records,
        "assigned_records": assigned_records,
        "start_pos": start_pos,
        "next_pos": next_pos,
        "processed_this_run": max(0, next_pos - start_pos),
        "reward_design_version": "chatjoy_format_role_language_length_v1",
        "counts": dict(counters),
        "started_at": started_at,
        "updated_at": time.time(),
        "completed": completed,
    }


def load_checkpoint(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_output(path: Path) -> Tuple[set[int], Counter, Counter, Counter, Counter, Counter, Counter, Counter, Counter, Counter]:
    seen_source_indices: set[int] = set()
    counters = Counter()
    target_role_counts = Counter()
    target_language_counts = Counter()
    wrong_role_source_role_counts = Counter()
    wrong_role_source_language_counts = Counter()
    wrong_language_source_role_counts = Counter()
    wrong_language_source_language_counts = Counter()
    length_source_role_counts = Counter()
    length_source_language_counts = Counter()
    if not path.exists():
        return (
            seen_source_indices,
            counters,
            target_role_counts,
            target_language_counts,
            wrong_role_source_role_counts,
            wrong_role_source_language_counts,
            wrong_language_source_role_counts,
            wrong_language_source_language_counts,
            length_source_role_counts,
            length_source_language_counts,
        )

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata = pair.get("metadata", {}) if isinstance(pair, dict) else {}
            source_index = metadata.get("source_index")
            if isinstance(source_index, int):
                seen_source_indices.add(source_index)
            rejected_type = pair.get("rejected_type")
            counters["written"] += 1
            if isinstance(rejected_type, str) and rejected_type:
                counters[rejected_type] += 1
            if rejected_type == "wrong_role":
                rejected_role = metadata.get("rejected_role")
                if isinstance(rejected_role, str) and rejected_role:
                    target_role_counts[rejected_role] += 1
                chosen_role = metadata.get("chosen_role")
                if isinstance(chosen_role, str) and chosen_role:
                    wrong_role_source_role_counts[chosen_role] += 1
                source_lang = str(pair.get("language", "")).strip().lower()
                if source_lang:
                    wrong_role_source_language_counts[source_lang] += 1
            elif rejected_type == "wrong_language":
                target_lang = metadata.get("target_wrong_language")
                if isinstance(target_lang, str) and target_lang:
                    target_language_counts[target_lang] += 1
                chosen_role = metadata.get("chosen_role")
                if isinstance(chosen_role, str) and chosen_role:
                    wrong_language_source_role_counts[chosen_role] += 1
                source_lang = str(pair.get("language", "")).strip().lower()
                if source_lang:
                    wrong_language_source_language_counts[source_lang] += 1
            elif isinstance(rejected_type, str) and rejected_type.startswith("length_mismatch"):
                chosen_role = metadata.get("chosen_role")
                if isinstance(chosen_role, str) and chosen_role:
                    length_source_role_counts[chosen_role] += 1
                source_lang = str(pair.get("language", "")).strip().lower()
                if source_lang:
                    length_source_language_counts[source_lang] += 1

    return (
        seen_source_indices,
        counters,
        target_role_counts,
        target_language_counts,
        wrong_role_source_role_counts,
        wrong_role_source_language_counts,
        wrong_language_source_role_counts,
        wrong_language_source_language_counts,
        length_source_role_counts,
        length_source_language_counts,
    )


def validate_checkpoint(args: argparse.Namespace, checkpoint: Dict, assigned_records: int) -> None:
    expected = {
        "input": args.input,
        "backend": args.backend,
        "mode": args.mode,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "input_ordering": args.input_ordering,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "wrong_role_attempts": args.wrong_role_attempts,
        "wrong_language_attempts": args.wrong_language_attempts,
        "length_attempts": args.length_attempts,
        "fallback_on_failure": args.fallback_on_failure,
        "fallback_attempts": args.fallback_attempts,
        "fallback_min_score_gap": args.fallback_min_score_gap,
        "fallback_dimension_gap_scale": args.fallback_dimension_gap_scale,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"Checkpoint mismatch for {key}: expected {value!r}, found {checkpoint.get(key)!r}. "
                "Use a fresh output path or pass consistent arguments."
            )
    checkpoint_assigned = checkpoint.get("assigned_records")
    if checkpoint_assigned is not None and checkpoint_assigned != assigned_records:
        raise ValueError(
            f"Checkpoint assigned_records mismatch: expected {assigned_records}, found {checkpoint_assigned}."
        )


def write_checkpoint(
    checkpoint_path: Path,
    *,
    args: argparse.Namespace,
    output_path: Path,
    report_path: Path,
    total_records: int,
    assigned_records: int,
    start_pos: int,
    next_pos: int,
    counters: Counter,
    started_at: float,
    completed: bool,
) -> None:
    payload = checkpoint_payload(
        args=args,
        output_path=output_path,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        total_records=total_records,
        assigned_records=assigned_records,
        start_pos=start_pos,
        next_pos=next_pos,
        counters=counters,
        started_at=started_at,
        completed=completed,
    )
    checkpoint_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    is_zero_gen = args.mode in ("cross_character", "cross_lingual")

    # Validate: model-path is required for model-gen modes
    if not is_zero_gen and not args.model_path:
        raise ValueError("--model-path is required when --backend is not 'none'")
    # Auto-set backend for zero-gen modes if not explicitly overridden
    if is_zero_gen and args.backend == "transformers":
        args.backend = "none"

    process_module = load_process_module(Path(args.process_script))
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be within [0, num_shards)")

    counters = Counter()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else output_path.with_suffix(
        output_path.suffix + ".state.json"
    )

    records = list(iter_records(Path(args.input)))
    total_records = len(records)
    if args.num_shards > 1:
        records = [record for pos, record in enumerate(records) if pos % args.num_shards == args.shard_index]
    if args.sample_size is not None:
        records = random.sample(records, min(args.sample_size, len(records)))
    assigned_records = len(records)
    role_catalog = load_role_catalog(Path(args.roles_dir))
    role_name_by_system = {r["system"]: r["role_id"] for r in role_catalog}
    records = reorder_records(records, role_name_by_system, args.input_ordering, args.seed)
    source_role_counts = count_source_roles(records, role_name_by_system)
    source_language_counts = count_source_languages(records)

    start_pos = 0
    started_at = time.time()
    existing_written_indices: set[int] = set()
    existing_counts = Counter()
    existing_target_role_counts = Counter()
    existing_target_language_counts = Counter()
    existing_wrong_role_source_role_counts = Counter()
    existing_wrong_role_source_language_counts = Counter()
    existing_wrong_language_source_role_counts = Counter()
    existing_wrong_language_source_language_counts = Counter()
    existing_length_source_role_counts = Counter()
    existing_length_source_language_counts = Counter()
    if args.resume:
        if output_path.exists():
            (
                existing_written_indices,
                existing_counts,
                existing_target_role_counts,
                existing_target_language_counts,
                existing_wrong_role_source_role_counts,
                existing_wrong_role_source_language_counts,
                existing_wrong_language_source_role_counts,
                existing_wrong_language_source_language_counts,
                existing_length_source_role_counts,
                existing_length_source_language_counts,
            ) = load_existing_output(output_path)
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint is not None:
            validate_checkpoint(args, checkpoint, assigned_records)
            counters.update(checkpoint.get("counts", {}))
            start_pos = int(checkpoint.get("next_pos", 0))
            started_at = float(checkpoint.get("started_at", started_at))
        elif args.skip_existing_output and output_path.exists():
            counters.update(existing_counts)
    elif output_path.exists() and not args.overwrite_output:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --resume to continue or --overwrite-output to replace it."
        )

    # Load cross-reference tables for zero-gen modes
    all_records_by_role: Dict[str, List] = {}
    all_records_by_lang_role: Dict[tuple, List] = {}
    if is_zero_gen:
        cross_ref_path = Path(args.all_records_path) if args.all_records_path else Path(args.input)
        for _, record in iter_records(cross_ref_path):
            role_id = role_name_by_system.get(str(record.get("system", "")).strip(), "unknown_role")
            lang = str(record.get("language", "unknown")).lower()
            all_records_by_role.setdefault(role_id, []).append(record)
            all_records_by_lang_role.setdefault((lang, role_id), []).append(record)

    builder = RejectedBuilder(
        model_path=Path(args.model_path) if args.model_path else Path("."),
        process_module=process_module,
        roles_dir=Path(args.roles_dir),
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        seed=args.seed,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        min_chosen_score=args.min_chosen_score,
        min_score_gap=args.min_score_gap,
        backend=args.backend,
        tokenizer_path=Path(args.tokenizer_path) if args.tokenizer_path else None,
        vllm_base_url=args.vllm_base_url,
        vllm_model=args.vllm_model,
        vllm_api_key=args.vllm_api_key,
        request_timeout=args.request_timeout,
        request_max_retries=args.request_max_retries,
        request_retry_sleep=args.request_retry_sleep,
        length_short_prob=args.length_short_prob,
        wrong_role_attempts=args.wrong_role_attempts,
        wrong_language_attempts=args.wrong_language_attempts,
        length_attempts=args.length_attempts,
        role_balance_mode=args.wrong_role_balance_mode,
        language_balance_mode=args.wrong_language_balance_mode,
        language_candidate_set=args.wrong_language_candidate_set,
        source_role_counts=source_role_counts,
        source_language_counts=source_language_counts,
        target_role_counts=existing_target_role_counts,
        target_language_counts=existing_target_language_counts,
        current_wrong_role_source_role_counts=existing_wrong_role_source_role_counts,
        current_wrong_role_source_language_counts=existing_wrong_role_source_language_counts,
        current_wrong_language_source_role_counts=existing_wrong_language_source_role_counts,
        current_wrong_language_source_language_counts=existing_wrong_language_source_language_counts,
        length_source_role_balance_mode=args.length_source_role_balance_mode,
        length_source_language_balance_mode=args.length_source_language_balance_mode,
        current_length_source_role_counts=existing_length_source_role_counts,
        current_length_source_language_counts=existing_length_source_language_counts,
        current_type_counts=existing_counts,
        fallback_on_failure=args.fallback_on_failure,
        fallback_attempts=args.fallback_attempts,
        fallback_min_score_gap=args.fallback_min_score_gap,
        fallback_dimension_gap_scale=args.fallback_dimension_gap_scale,
        all_records_by_role=all_records_by_role,
        all_records_by_lang_role=all_records_by_lang_role,
    )

    if start_pos > assigned_records:
        raise ValueError(
            f"Checkpoint next_pos {start_pos} exceeds assigned_records {assigned_records} for {checkpoint_path}"
        )

    file_mode = "a" if args.resume and output_path.exists() else "w"
    pending_records = [
        (index, item)
        for index, item in records[start_pos:]
        if index not in existing_written_indices
    ]

    with output_path.open(file_mode, encoding="utf-8") as out_f:
        for local_processed, (index, item) in enumerate(pending_records, start=1):
            processed = start_pos + local_processed

            if args.mode == "cross_character":
                pair = builder.build_cross_character_pair(index, item)
                target_type = "cross_character"
            elif args.mode == "cross_lingual":
                pair = builder.build_cross_lingual_pair(index, item)
                target_type = "cross_lingual"
            elif args.mode == "mixed":
                target_type = builder.choose_type_for_item(
                    item,
                    args.wrong_role_weight,
                    args.wrong_language_weight,
                    args.length_weight,
                )
                pair = builder.build_pair(index, item, target_type, max_attempts=args.max_attempts)
            else:
                target_type = args.mode
                pair = builder.build_pair(index, item, target_type, max_attempts=args.max_attempts)

            if pair is None:
                counters["failed"] += 1
                counters[f"failed_{target_type}"] += 1
            else:
                out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                out_f.flush()
                counters["written"] += 1
                counters[pair["rejected_type"]] += 1

            if args.checkpoint_every and processed % args.checkpoint_every == 0:
                write_checkpoint(
                    checkpoint_path,
                    args=args,
                    output_path=output_path,
                    report_path=report_path,
                    total_records=total_records,
                    assigned_records=assigned_records,
                    start_pos=start_pos,
                    next_pos=processed,
                    counters=counters,
                    started_at=started_at,
                    completed=False,
                )

            if args.log_every and processed % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "assigned_records": assigned_records,
                            "written": counters["written"],
                            "failed": counters["failed"],
                            "backend": args.backend,
                            "shard_index": args.shard_index,
                            "num_shards": args.num_shards,
                            "checkpoint": str(checkpoint_path),
                            "skipped_existing_output": len(existing_written_indices),
                        },
                        ensure_ascii=False,
                    )
                )

    write_checkpoint(
        checkpoint_path,
        args=args,
        output_path=output_path,
        report_path=report_path,
        total_records=total_records,
        assigned_records=assigned_records,
        start_pos=start_pos,
        next_pos=assigned_records,
        counters=counters,
        started_at=started_at,
        completed=True,
    )

    report = {
        "input": args.input,
        "output": str(output_path),
        "report": str(report_path),
        "checkpoint": str(checkpoint_path),
        "backend": args.backend,
        "mode": args.mode,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "input_ordering": args.input_ordering,
        "max_attempts": args.max_attempts,
        "wrong_role_attempts": args.wrong_role_attempts,
        "wrong_language_attempts": args.wrong_language_attempts,
        "length_attempts": args.length_attempts,
        "min_chosen_score": args.min_chosen_score,
        "min_score_gap": args.min_score_gap,
        "fallback_on_failure": args.fallback_on_failure,
        "fallback_attempts": args.fallback_attempts,
        "fallback_min_score_gap": args.fallback_min_score_gap,
        "fallback_dimension_gap_scale": args.fallback_dimension_gap_scale,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "input_records_total": total_records,
        "assigned_records": assigned_records,
        "resume_enabled": args.resume,
        "skipped_existing_output": len(existing_written_indices),
        "reward_design_version": "chatjoy_format_role_language_length_v1",
        "counts": dict(counters),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
