#!/usr/bin/env python3
"""
针对 v6.9 退化的 attack 用例生成补充 defense 训练数据。
用法: python3 gen_defense_supplement.py --out defense_supplement.jsonl --n 8
"""
import argparse, json, sys, time, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, "/root/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package")
from anti_ooc_disclosure import detect_ooc_disclosure

# 需要补充的用例:v6.9 退化但 step0 没退化(即可学的),优先 A6 code_block + ML 多语言
FOCUS_CASES = {
    "A6",                          # task injection → code_block (最大缺口)
    "ML_ja_A6", "ML_ko_A6", "ML_zh-tw_A6", "ML_sk_A6", "ML_en_A6",
    "ML_ja_A1", "ML_ko_A1", "ML_zh-tw_A1", "ML_sk_A1", "ML_zh_A1",
    "A2", "A3",                    # zh roledrop variants
    "A1_ja",
}

def load_attack_cases(fn="/root/rich_cjk_sft_v69/step0_attack_t0.9.jsonl"):
    rows = [json.loads(l) for l in open(fn) if l.strip()]
    # deduplicate by (case_id, lang, user)
    seen = set()
    out = []
    for r in rows:
        k = (r["case_id"], r.get("lang",""), r["user"][:60])
        if k not in seen and r["case_id"] in FOCUS_CASES:
            seen.add(k)
            out.append(r)
    return out


def post_chat(url, model, system, user, history=None, retries=4):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    for u, a in (history or []):
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user})
    payload = {
        "model": model, "messages": messages,
        "temperature": 0.95, "top_p": 0.95,
        "presence_penalty": 0.4, "max_tokens": 768,
        "truncate_prompt_tokens": 3072,
    }
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
    ap.add_argument("--out", default="defense_supplement.jsonl")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="step0")
    ap.add_argument("--n", type=int, default=8, help="candidates per case")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    cases = load_attack_cases()
    print(f"[load] {len(cases)} unique attack cases to supplement")

    out_path = Path(a.out)
    done_keys = set()
    written = 0
    if a.resume and out_path.exists():
        for line in out_path.open():
            if line.strip():
                r = json.loads(line)
                done_keys.add((r.get("case_id"), r.get("lang"), r.get("instruction","")[:60]))
                written += 1
        print(f"[resume] {written} already written")

    good = bad = 0
    with out_path.open("a" if a.resume else "w", encoding="utf-8") as f:
        for i, case in enumerate(cases):
            k = (case["case_id"], case.get("lang",""), case["user"][:60])
            generated_for_case = 0
            for attempt in range(a.n):
                response = post_chat(a.url, a.model, case.get("system",""), case["user"])
                is_ooc, reason = detect_ooc_disclosure(response)
                if is_ooc:
                    bad += 1
                    print(f"  [skip OOC] {case['case_id']} {case.get('lang','')} reason={reason}", flush=True)
                    continue
                # Good: stays in character
                good += 1
                generated_for_case += 1
                rec = {
                    "system":      case.get("system", ""),
                    "instruction": case["user"],
                    "input":       "",
                    "output":      response,
                    "history":     [],
                    "meta": {
                        "case_id": case["case_id"],
                        "lang":    case.get("lang", ""),
                        "char":    case.get("char", ""),
                        "source":  "defense_supplement_v69",
                    },
                    "richness_score": 0.0,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            print(f"[{i+1}/{len(cases)}] {case['case_id']} {case.get('lang','')} → kept {generated_for_case}/{a.n}", flush=True)

    total = good + bad
    print(f"\nDone: {good} kept / {bad} filtered OOC / total generated {total}")
    print(f"Output: {out_path} ({good} records)")


if __name__ == "__main__":
    main()
