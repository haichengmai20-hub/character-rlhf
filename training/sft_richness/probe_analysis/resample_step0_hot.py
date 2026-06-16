#!/usr/bin/env python3
"""resample_step0_hot.py
用高温度采样重采 step0 模型的 110 条 probe，对比原始低温采样。

采样参数对比:
  原始: temp=0.7, top_p=0.8, max_tokens=768, seed=42
  高温: temp=1.0, top_p=0.95, presence_penalty=0.4, max_tokens=768

用法:
  /home/ubuntu/miniconda3/envs/gemma4-rl/bin/python3 resample_step0_hot.py
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
    ap.add_argument("--model", default="/root/sft_rich_cjk_v67_repair_merged_full")
    ap.add_argument("--probe", default="/root/outputs/v67_step0_vs_step100/step0_probe.jsonl")
    ap.add_argument("--out", default="/root/outputs/v67_step0_vs_step100/step0_probe_hot.jsonl")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--presence-penalty", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    # 加载 probe 数据
    records = load_probe(a.probe)
    print(f"[load] {len(records)} probe records")

    # 构建 vLLM prompts — 用 chat template 渲染 system + user
    from transformers import AutoTokenizer
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

    print(f"[info] {len(prompts)} prompts, temp={a.temp} top_p={a.top_p} presence_penalty={a.presence_penalty}")

    # 加载模型
    t0 = time.time()
    llm = LLM(
        model=a.model,
        tensor_parallel_size=a.tp,
        gpu_memory_utilization=a.gpu_mem,
        max_model_len=a.max_model_len,
        limit_mm_per_prompt={"image": 0, "video": 0},
        trust_remote_code=True,
        enforce_eager=True,
    )
    print(f"[load] model loaded in {time.time()-t0:.1f}s")

    # 采样参数
    sp = SamplingParams(
        temperature=a.temp,
        top_p=a.top_p,
        presence_penalty=a.presence_penalty,
        max_tokens=a.max_tokens,
    )

    # 生成
    t1 = time.time()
    outputs = llm.generate(prompts, sp)
    print(f"[gen] generated {len(outputs)} in {time.time()-t1:.1f}s")

    # 写结果 — 和原始 probe 格式一致，替换 response + sampling
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
            new_rec["model"] = rec.get("model", "step0_s1") + "_hot"
            f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")

    print(f"[done] wrote {len(records)} -> {a.out}")

    # 快速统计 response 长度
    lens = [len(out.outputs[0].text) for out in outputs]
    arr = np.array(lens)
    print(f"\n=== Response Length ===")
    print(f"  mean={arr.mean():.0f} std={arr.std():.0f} min={arr.min()} max={arr.max()}")

    # 对比原始 response 长度
    orig_lens = [len(rec.get("response", "")) for rec in records]
    orig_arr = np.array(orig_lens)
    print(f"\n=== Original Response Length ===")
    print(f"  mean={orig_arr.mean():.0f} std={orig_arr.std():.0f} min={orig_arr.min()} max={orig_arr.max()}")


if __name__ == "__main__":
    main()
