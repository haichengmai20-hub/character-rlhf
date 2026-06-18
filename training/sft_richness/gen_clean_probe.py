#!/usr/bin/env python3
"""
Generate base_it or step0 responses on the seed=42 200-prompt clean probe.
Usage:
  python3 gen_clean_probe.py --model base_it --url http://127.0.0.1:8000 --out base_it_clean200.jsonl
  python3 gen_clean_probe.py --model step0   --url http://127.0.0.1:8001 --out step0_clean200.jsonl
"""
import argparse, json, random, re, time, urllib.error, urllib.request
from collections import defaultdict
from pathlib import Path

PROMPTS = Path("/root/rich_cjk_sft_v67/prompts_base_teacher.jsonl")
LANGS = ("zh", "zh-tw", "ja", "ko")
N_PER_LANG = 50

# Dialog regex: opening = CJK corner brackets + curly double/single quotes (unicode-escaped to avoid Py3.12 SyntaxError)
# 「 = 「, “ = ", ‘ = ', 『 = 『
# 」 = 」, ” = ", ’ = ', 』 = 』
_DLG_RE = re.compile(
    "[「“‘『][^”’』」\n]+[”’」』]"
)
_ACT_RE   = re.compile(r"\*[^*\n]{2,}\*")
_PAREN_RE = re.compile(r"（[^）]{2,}）")  # （...）full-width paren


def seed42_sample(seed=42):
    buckets = defaultdict(list)
    for line in PROMPTS.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("language") in LANGS and row.get("messages"):
            buckets[row["language"]].append(row)
    random.seed(seed)
    selected = []
    for lang in LANGS:
        selected.extend(random.sample(buckets[lang], N_PER_LANG))
    return selected


def last_assistant_index(messages):
    idx = None
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            idx = i
    return idx


def build_probe_messages(row):
    msgs = row["messages"]
    idx = last_assistant_index(msgs)
    return msgs[:idx] if idx is not None else msgs


def metrics(text):
    return {
        "action_blocks": len(_ACT_RE.findall(text)),
        "paren_blocks":  len(_PAREN_RE.findall(text)),
        "dialog_blocks": len(_DLG_RE.findall(text)),
        "has_monologue": bool(_PAREN_RE.search(text)),
        "char_count":    len(text),
    }


def post_chat(url, model_name, messages, temp=0.9, top_p=0.95, presence=0.4, min_p=0.0, max_tokens=1024, retries=4):
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temp,
        "top_p": top_p,
        "presence_penalty": presence,
        "max_tokens": max_tokens,
        "truncate_prompt_tokens": 3072,
    }
    if min_p > 0:
        payload["min_p"] = min_p
    data = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(retries):
        try:
            with opener.open(req, timeout=180) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"request failed: {e}") from e
            time.sleep(2 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="served-model-name (e.g. base_it)")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--presence", type=float, default=0.4)
    ap.add_argument("--min-p", type=float, default=0.0, help="min_p sampling parameter (0=disabled)")
    ap.add_argument("--max-tokens", type=int, default=None)  # None = use script default 1024
    ap.add_argument("--seed", type=int, default=42, help="random seed for prompt sampling")
    a = ap.parse_args()
    _max_tokens = a.max_tokens if a.max_tokens is not None else 1024

    selected = seed42_sample(seed=a.seed)
    out_path = Path(a.out)
    done_ids = set()
    if a.resume and out_path.exists():
        for line in out_path.open():
            if line.strip():
                done_ids.add(json.loads(line).get("prompt_id"))
        print(f"[resume] {len(done_ids)} already done")

    with out_path.open("a" if a.resume else "w", encoding="utf-8") as f:
        for i, row in enumerate(selected, 1):
            pid = row.get("prompt_id")
            if pid in done_ids:
                continue
            lang = row["language"]
            msgs = build_probe_messages(row)
            response = post_chat(a.url, a.model, msgs, temp=a.temp, top_p=a.top_p, presence=a.presence, min_p=a.min_p, max_tokens=_max_tokens)
            item = {"prompt_id": pid, "lang": lang, "model": a.model, **metrics(response), "response": response}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/200] {lang} {pid} act={item['action_blocks']} dlg={item['dialog_blocks']} mono={item['has_monologue']}", flush=True)


if __name__ == "__main__":
    main()
