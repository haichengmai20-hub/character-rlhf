"""过滤 teacher 生成(去身份泄露/拒绝/无格式)→ 拒绝采样取每 prompt 最优 → 配**原始 system**
成 SFT pair → 混 replay 防遗忘 → 写 LlamaFactory(alpaca)json。

输入 prompts_hinted.jsonl 每行带 :rK 后缀(同一 base prompt 的多次采样)、orig_system、language。
gen_raw.jsonl 每行 response 对应一个 :rK prompt_id。
"""
import argparse
import json
import random
import re

from anti_ooc_disclosure import detect_ooc_disclosure
from anti_refusal_reward import detect_refusal, _has_roleplay_format


def keep_response(resp: str):
    """戏内化解才留:非身份泄露(多语言) + 非拒绝 + 有 roleplay 格式。"""
    resp = (resp or "").strip()
    if not resp:
        return False, "empty"
    is_disc, _ = detect_ooc_disclosure(resp)
    if is_disc:
        return False, "disclosure"
    is_ref, _ = detect_refusal(resp)          # detect_refusal 返回 (bool, reason)
    if is_ref:
        return False, "refusal"
    if not _has_roleplay_format(resp):
        return False, "no_format"
    return True, "ok"


# 非拉丁语种:脚本检测(稳健,避开 langdetect 对短 CJK 的混淆;主要抓"漂移成英文")
_SCRIPT_PATTERNS = {
    "zh": r"[一-鿿]", "zh-tw": r"[一-鿿]",
    "ja": r"[぀-ヿ一-鿿]",   # kana + 汉字
    "ko": r"[가-힯]",
    "ru": r"[Ѐ-ӿ]", "uk": r"[Ѐ-ӿ]",
    "th": r"[฀-๿]",
    "ar": r"[؀-ۿ]", "fa": r"[؀-ۿ]",
}


def _script_frac(text: str, pattern: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if re.match(pattern, c)) / len(letters)


def lang_matches(output: str, expected_lang: str) -> bool:
    """output 语言是否与攻击语言一致(防 teacher 在攻击下漂移成英文)。
    非拉丁语种用脚本占比(>=30%);拉丁语种用 langdetect(期望语言须在 top-2)。
    """
    if not expected_lang:
        return True
    exp = expected_lang.lower()
    text = (output or "").strip()
    if exp in _SCRIPT_PATTERNS:
        return _script_frac(text, _SCRIPT_PATTERNS[exp]) >= 0.30
    from langdetect import detect_langs, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0
    try:
        cands = detect_langs(text)
    except LangDetectException:
        return False
    return any(c.lang.lower() == exp for c in cands[:2])


def select_best(candidates, expected_lang=None):
    """从同一 prompt 的多次采样里,选通过过滤(+语言一致)且**最长**(最沉浸)的一条;都不过返回 None。"""
    passing = [c.strip() for c in candidates if keep_response(c)[0]]
    if expected_lang:
        passing = [c for c in passing if lang_matches(c, expected_lang)]
    if not passing:
        return None
    return max(passing, key=len)


def to_sft_row(orig_system: str, user: str, response: str) -> dict:
    return {"system": orig_system, "instruction": user, "input": "",
            "output": response, "history": []}


def _base_pid(pid: str) -> str:
    return pid.rsplit(":r", 1)[0] if ":r" in pid else pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="+", required=True, help="一个或多个 hinted prompts 文件")
    ap.add_argument("--gen", nargs="+", required=True, help="对应的一个或多个 gen_raw 文件")
    ap.add_argument("--replay", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replay-mult", type=float, default=4.0)
    ap.add_argument("--resp-key", default="response")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # base prompt_id -> (orig_system, user, language)
    pmap = {}
    for path in args.prompts:
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            bpid = _base_pid(r["prompt_id"])
            if bpid in pmap:
                continue
            user = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
            pmap[bpid] = (r.get("orig_system", ""), user, r.get("language"))

    # 收集每个 base prompt 的所有采样回答
    cands = {}
    for path in args.gen:
        for line in open(path):
            if not line.strip():
                continue
            g = json.loads(line)
            bpid = _base_pid(g.get("prompt_id", ""))
            resp = g.get(args.resp_key) or g.get("response") or ""
            cands.setdefault(bpid, []).append(resp)

    kept, by_lang = [], {}
    for bpid, (sys_, user, lang) in pmap.items():
        best = select_best(cands.get(bpid, []), expected_lang=lang)
        if best is None:
            continue
        kept.append(to_sft_row(sys_, user, best))
        by_lang[lang] = by_lang.get(lang, 0) + 1

    random.seed(args.seed)
    replay_all = json.load(open(args.replay))
    n_replay = min(len(replay_all), int(len(kept) * args.replay_mult))
    replay = random.sample(replay_all, n_replay)

    data = kept + replay
    random.shuffle(data)
    json.dump(data, open(args.out, "w"), ensure_ascii=False)
    print(f"[defense-sft] kept={len(kept)} by_lang={by_lang} "
          f"prompts_covered={len(kept)}/{len(pmap)} replay={n_replay} total={len(data)} -> {args.out}")


if __name__ == "__main__":
    main()
