#!/usr/bin/env python3
"""
把 rm_train_weighted_{train,eval}.jsonl 转换为 OpenRLHF RM 格式:
  {
    "prompt":   [{"role":"system","content":"..."}, {"role":"user","content":"..."}],
    "chosen":   [{"role":"assistant","content":"..."}],
    "rejected": [{"role":"assistant","content":"..."}],
    "extra_info": {"language":..., "rejected_type":..., "reward_gap":..., "sample_weight":...}
  }
"""
import json
from pathlib import Path

DATA_DIR  = Path("/home/ps/hanjiaqi/LlamaFactory/chatjoy/data")
OUT_DIR   = Path("/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/openrlhf_rm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for split in ("train", "eval"):
    src  = DATA_DIR / f"rm_train_weighted_{split}.jsonl"
    dst  = OUT_DIR  / f"rm_{split}.jsonl"
    n    = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            r = json.loads(line)
            prompt_messages = []
            if r.get("system"):
                prompt_messages.append({"role": "system", "content": r["system"]})
            prompt_messages.append({"role": "user", "content": r["prompt"]})

            out = {
                "prompt":   prompt_messages,
                "chosen":   [{"role": "assistant", "content": r["chosen"]}],
                "rejected": [{"role": "assistant", "content": r["rejected"]}],
                "extra_info": {
                    "language":      r.get("language", ""),
                    "rejected_type": r.get("rejected_type", ""),
                    "reward_gap":    r.get("reward_gap", 0.0),
                    "sample_weight": r.get("sample_weight", 1.0),
                    "chosen_role":   r.get("chosen_role", ""),
                },
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print(f"{split}: {n} records → {dst}")
