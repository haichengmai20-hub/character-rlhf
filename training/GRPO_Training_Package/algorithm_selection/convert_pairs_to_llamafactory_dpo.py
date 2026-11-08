#!/usr/bin/env python
"""RM pairdata（rm_v6_finegrained/build_pairs.py 产物）-> LLaMA-Factory ranking(DPO) 格式转换。

输入 schema（build_pairs.py L37-44 口径，每行一对）：
    {"prompt": [ {role, content}, ... ],        # 完整多轮前缀 messages
     "chosen":  {"role": "assistant", "content": "..."},
     "rejected": {"role": "assistant", "content": "..."},
     "prompt_id": "...", ...其余元数据原样保留在 --keep-meta 指定的键中}

输出（LLaMA-Factory sharegpt 偏好格式）：
    {"messages": prompt, "chosen": chosen, "rejected": rejected}

处理规则：
- 缺 prompt/chosen/rejected、空 content、prompt 非列表 -> 跳过并计数；
- 以 prompt_id（缺省用 chosen+rejected 内容哈希）去重；
- --shuffle/--limit 做平衡抽样的入口（分层抽样在数据侧完成后再传入）；
- 不改写任何文本；raw/summary 输入协议的转换在上游完成后才可喂入。

用法：
    python convert_pairs_to_llamafactory_dpo.py \
        --input pairs.jsonl --output data/character_rlhf_dpo.jsonl \
        --emit-dataset-info data/dataset_info.json \
        [--shuffle --seed 11 --limit 0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random


def content_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def valid_pair(row: dict) -> bool:
    p, c, r = row.get("prompt"), row.get("chosen"), row.get("rejected")
    if not isinstance(p, list) or not p:
        return False
    for side in (c, r):
        if not isinstance(side, dict):
            return False
        content = side.get("content")
        if not isinstance(content, str) or not content.strip():
            return False
    return True


def convert(rows: list[dict], shuffle: bool, seed: int, limit: int):
    seen, out, skipped = set(), [], {"invalid": 0, "duplicate": 0}
    for row in rows:
        if not valid_pair(row):
            skipped["invalid"] += 1
            continue
        key = str(row.get("prompt_id") or content_hash([row["chosen"], row["rejected"]]))
        if key in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(key)
        out.append({"messages": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]})
    if shuffle:
        random.Random(seed).shuffle(out)
    if limit and limit > 0:
        out = out[:limit]
    return out, skipped


DATASET_INFO_TEMPLATE = {
    "character_rlhf_dpo": {
        "file_name": "character_rlhf_dpo.jsonl",
        "ranking": True,
        "format": "sharegpt",
        "columns": {"messages": "messages", "chosen": "chosen", "rejected": "rejected"},
    }
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--emit-dataset-info", default=None, help="同时生成 dataset_info.json 起点模板")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--limit", type=int, default=0, help="0 = 不截断（开放完整偏好池）")
    args = ap.parse_args(argv)

    with open(args.input, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    out, skipped = convert(rows, args.shuffle, args.seed, args.limit)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.emit_dataset_info:
        with open(args.emit_dataset_info, "w", encoding="utf-8") as f:
            json.dump(DATASET_INFO_TEMPLATE, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print(json.dumps({
        "input_rows": len(rows), "output_pairs": len(out),
        "skipped": skipped, "output": args.output,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
