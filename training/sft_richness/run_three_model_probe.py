#!/usr/bin/env python3
"""v6.9 Three-model clean+attack probe (同口径)
- Track A: base_it + step0 clean probe (200条, seed=42)
- Track B: v6.9 attack probe (68条 A用例, t0.9)
v6.9 clean 已有,只需重算regex
"""
import json, re, os, time, random, sys
import urllib.request
from collections import Counter

# === 统一 regex ===
RE_ACTION = r'\*[^*\n]{2,}\*'
RE_DIALOG = r'[「『\u201c\u201d"][^」』\u201c\u201d"]+[」』\u201c\u201d"]'
RE_MONO = r'（[^）]{2,}）'
RE_PAREN = r'（[^）]{2,}）'

def score_response(resp):
    return {
        "action": len(re.findall(RE_ACTION, resp)),
        "dialog": len(re.findall(RE_DIALOG, resp)),
        "mono_bool": bool(re.search(RE_MONO, resp)),
        "paren_count": len(re.findall(RE_PAREN, resp)),
        "chars": len(resp),
    }

# === Config ===
VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
TEMP = 0.9
TOP_P = 0.95
PRESENCE = 0.4
MAXTOK = 1024
TRUNCATE = 3072
NSAMPLES = 50

os.environ['no_proxy'] = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

def call_vllm(messages, model_name, max_retries=3):
    payload = json.dumps({
        "model": model_name,
        "messages": messages,
        "temperature": TEMP,
        "top_p": TOP_P,
        "presence_penalty": PRESENCE,
        "max_tokens": MAXTOK,
        "truncate_prompt_tokens": TRUNCATE,
    }).encode()
    req = urllib.request.Request(VLLM_URL, data=payload, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(max_retries):
        try:
            with opener.open(req, timeout=120) as resp:
                result = json.loads(resp.read().decode())
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return f"ERROR: {e}"

def build_probe_messages(p):
    """From prompts_base_teacher format: strip last assistant turn"""
    messages = p.get("messages", [])
    probe_msgs = []
    for m in messages:
        if m["role"] == "system":
            probe_msgs.append({"role": "system", "content": m["content"]})
            break
    last_asst = -1
    for i in range(len(messages)-1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_asst = i
            break
    end = last_asst if last_asst > 0 else len(messages)
    for m in messages[1:end]:
        if m["role"] != "system":
            probe_msgs.append({"role": m["role"], "content": m["content"]})
    return probe_msgs

# === Load prompts ===
with open("/root/rich_cjk_sft_v67/prompts_base_teacher.jsonl") as f:
    all_prompts = [json.loads(l) for l in f]

by_lang = {}
for p in all_prompts:
    lang = p.get("language", "")
    if lang in ("zh", "zh-tw", "ja", "ko"):
        by_lang.setdefault(lang, []).append(p)

random.seed(42)
test_set = {}
for lang in ("zh", "zh-tw", "ja", "ko"):
    ps = by_lang.get(lang, [])
    test_set[lang] = random.sample(ps, min(NSAMPLES, len(ps)))

# Collect all 200 prompts in order
all_test = []
for lang in ("zh", "zh-tw", "ja", "ko"):
    for p in test_set[lang]:
        all_test.append((lang, p))

print(f"Clean probe: {len(all_test)} prompts")

# === Load attack prompts ===
with open("/root/rich_cjk_sft_v69/attack_prompts.jsonl") as f:
    attack_prompts = [json.loads(l) for l in f]
print(f"Attack probe: {len(attack_prompts)} prompts")

# === MODE ===
# "base_it" or "step0" or "v69_attack"
mode = sys.argv[1] if len(sys.argv) > 1 else "base_it"

if mode == "base_it":
    model_name = "base_it"
    vllm_model = "base_it"
    out_clean = "/root/rich_cjk_sft_v69/base_it_clean_probe.jsonl"
    out_attack = None
elif mode == "step0":
    model_name = "step0"
    vllm_model = "step0"
    out_clean = "/root/rich_cjk_sft_v69/step0_clean_probe.jsonl"
    out_attack = None
elif mode == "v69_attack":
    model_name = "v69"
    vllm_model = "v69"
    out_clean = None
    out_attack = "/root/rich_cjk_sft_v69/v69_attack_probe.jsonl"
else:
    print(f"Unknown mode: {mode}")
    sys.exit(1)

# vLLM model name override via env
if os.environ.get("VLLM_MODEL_NAME"):
    vllm_model = os.environ["VLLM_MODEL_NAME"]

# === Run clean probe ===
if out_clean:
    print(f"\n=== Running clean probe for {model_name} ===")
    clean_results = []
    for i, (lang, p) in enumerate(all_test):
        probe_msgs = build_probe_messages(p)
        resp = call_vllm(probe_msgs, vllm_model)
        if resp.startswith("ERROR"):
            print(f"  ERROR at {i}: {resp[:80]}")
            continue
        scores = score_response(resp)
        clean_results.append({
            "model": model_name, "lang": lang,
            "prompt_id": str(p.get("prompt_id", i)),
            **scores, "response": resp,
        })
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(all_test)} done")

    with open(out_clean, "w") as f:
        for r in clean_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(clean_results)} -> {out_clean}")

# === Run attack probe ===
if out_attack:
    print(f"\n=== Running attack probe for {model_name} ===")
    attack_results = []
    for i, ap in enumerate(attack_prompts):
        messages = []
        if ap.get("system"):
            messages.append({"role": "system", "content": ap["system"]})
        if ap.get("user"):
            messages.append({"role": "user", "content": ap["user"]})
        
        resp = call_vllm(messages, vllm_model)
        if resp.startswith("ERROR"):
            print(f"  ERROR at {i}: {resp[:80]}")
            continue
        
        attack_results.append({
            "model": model_name,
            "case_id": ap["case_id"],
            "lang": ap.get("lang", ""),
            "char": ap.get("char", ""),
            "system": ap.get("system", ""),
            "user": ap.get("user", ""),
            "response": resp,
        })
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(attack_prompts)} done")

    with open(out_attack, "w") as f:
        for r in attack_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(attack_results)} -> {out_attack}")

print("\nDone!")
