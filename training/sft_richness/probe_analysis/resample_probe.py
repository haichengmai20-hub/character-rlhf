#!/usr/bin/env python3
"""resample_probe.py
通用 probe 重采脚本 — 指定模型 + 采样参数,对 probe jsonl 重采并保存。

用法:
  /home/ubuntu/miniconda3/envs/gemma4-rl/bin/python3 resample_probe.py \
    --model /root/sft_rich_cjk_v67_repair_merged_full \
    --probe /root/outputs/v67_step0_vs_step100/step0_probe.jsonl \
    --out /root/outputs/v67_step0_vs_step100/step0_t0.8.jsonl \
    --temp 0.8 --top-p 0.95 --presence-penalty 0.4
"""
import argparse, json, os, time
import numpy as np


def load_probe(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--presence-penalty", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    records = load_probe(a.probe)
    print(f"[load] {len(records)} probe records from {a.probe}")

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    prompts = []
    for rec in records:
        msgs = []
        if rec.get("system"):
            msgs.append({"role": "system", "content": rec["system"]})
        if rec.get("user"):
            msgs.append({"role": "user", "content": rec["user"]})
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(rendered)

    print(f"[info] {len(prompts)} prompts, temp={a.temp} top_p={a.top_p} pp={a.presence_penalty}")

    t0 = time.time()
    llm = LLM(
        model=a.model, tensor_parallel_size=a.tp,
        gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len,
        limit_mm_per_prompt={"image": 0, "video": 0},
        trust_remote_code=True, enforce_eager=True,
    )
    print(f"[load] model loaded in {time.time()-t0:.1f}s")

    sp = SamplingParams(
        temperature=a.temp, top_p=a.top_p,
        presence_penalty=a.presence_penalty, max_tokens=a.max_tokens,
    )

    t1 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t1
    print(f"[gen] generated {len(outputs)} in {elapsed:.1f}s")

    with open(a.out, "w") as f:
        for rec, out in zip(records, outputs):
            generated_text = out.outputs[0].text
            new_rec = dict(rec)
            new_rec["response"] = generated_text
            new_rec["sampling"] = {
                "temperature": a.temp,
                "top_p": a.top_p,
                "presence_penalty": a.presence_penalty,
                "max_tokens": a.max_tokens,
            }
            # 保留原始 model 字段但加温度后缀
            orig_model = rec.get("model", "unknown")
            new_rec["model"] = f"{orig_model}_t{a.temp}"
            f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")

    lens = [len(out.outputs[0].text) for out in outputs]
    arr = np.array(lens)
    print(f"[done] wrote {len(records)} -> {a.out}")
    print(f"  response len: mean={arr.mean():.0f} std={arr.std():.0f} min={arr.min()} max={arr.max()}")


if __name__ == "__main__":
    main()
