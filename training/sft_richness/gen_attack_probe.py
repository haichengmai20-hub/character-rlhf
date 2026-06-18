#!/usr/bin/env python3
"""
用已有的 attack 用例(step0_t0.9.jsonl 的 attack rows)向 v6.9 生成 attack 回应。
用法:
  python3 gen_attack_probe.py --model v69 --url http://127.0.0.1:8000 --out v69_attack.jsonl
"""
import argparse, json, time, urllib.error, urllib.request
from pathlib import Path

STEP0_T09 = Path("/root/outputs/v67_step0_vs_step100/step0_t0.9.jsonl")


def load_attack_cases():
    rows = [json.loads(l) for l in STEP0_T09.open() if l.strip()]
    return [r for r in rows if "A" in r.get("case_id", "")]


def post_chat(url, model_name, system, user, temp=0.9, top_p=0.95, presence=0.4, min_p=0.0, max_tokens=768, retries=4):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if user:
        messages.append({"role": "user", "content": user})
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
    req = urllib.request.Request(url + "/v1/chat/completions", data=data,
                                  headers={"Content-Type": "application/json"}, method="POST")
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--presence", type=float, default=0.4)
    ap.add_argument("--min-p", type=float, default=0.0, help="min_p sampling parameter (0=disabled)")
    ap.add_argument("--max-tokens", type=int, default=None)  # None = use script default 768
    a = ap.parse_args()
    _max_tokens = a.max_tokens if a.max_tokens is not None else 768

    cases = load_attack_cases()
    print(f"[load] {len(cases)} attack cases")

    out_path = Path(a.out)
    done_ids = set()
    if a.resume and out_path.exists():
        for line in out_path.open():
            if line.strip():
                r = json.loads(line)
                done_ids.add((r.get("case_id"), r.get("lang")))

    with out_path.open("a" if a.resume else "w", encoding="utf-8") as f:
        for i, case in enumerate(cases, 1):
            key = (case["case_id"], case["lang"])
            if key in done_ids:
                continue
            response = post_chat(a.url, a.model, case.get("system", ""), case.get("user", ""),
                                 temp=a.temp, top_p=a.top_p, presence=a.presence, min_p=a.min_p, max_tokens=_max_tokens)
            out_rec = {
                "char":     case.get("char", ""),
                "char_id":  case.get("char_id", ""),
                "case_id":  case["case_id"],
                "lang":     case.get("lang", ""),
                "model":    a.model,
                "system":   case.get("system", ""),
                "user":     case.get("user", ""),
                "response": response,
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(cases)}] {case['case_id']} {case.get('lang','?')}", flush=True)


if __name__ == "__main__":
    main()
