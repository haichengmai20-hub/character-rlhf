#!/usr/bin/env python3
"""v6.9 best-of-N resampling: base IT × N=8 per CJK scene, NEUTRAL system (no style_hint).

Uses vLLM offline batch generation with apply_chat_template.
8-GPU TP=8 for maximum throughput.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--temp", type=float, default=0.95)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--presence-penalty", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--lang", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2000, help="Prompts per batch")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N already-processed inputs (for resume)")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # Load prompts
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if args.lang and r.get('language') not in args.lang:
                continue
            prompts.append(r)

    if args.limit:
        prompts = prompts[:args.limit]

    print(f"[init] {len(prompts)} prompts × N={args.n} = {len(prompts)*args.n} total generations")

    # Build metadata: N copies per prompt
    all_meta = []
    for p in prompts:
        system = p.get('sft_system', '') or p.get('system_prompt', '') or ''
        # ★ NEUTRAL: NO style_hint (v68 monologue over-shoot root cause)
        instruction = p.get('sft_instruction', '') or p.get('instruction', '')
        for n_idx in range(args.n):
            all_meta.append({
                "prompt_id": p.get('prompt_id', ''),
                "language": p.get('language', ''),
                "source": f"base_it_bon_n{n_idx}",
                "n_idx": n_idx,
                "system": system,
                "instruction": instruction,
            })

    # Skip already-processed inputs (for resume)
    if args.skip > 0:
        print(f"[resume] skipping first {args.skip} already-processed inputs")
        all_meta = all_meta[args.skip:]

    print(f"[init] {len(all_meta)} total inputs ({'remaining' if args.skip else 'total'})")

    # Init vLLM + tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    print(f"[load] model loaded in {time.time()-t0:.1f}s")

    sampling = SamplingParams(
        temperature=args.temp,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        max_tokens=args.max_tokens,
    )

    # Process in batches
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    all_lens = []

    for batch_start in range(0, len(all_meta), args.batch_size):
        batch_meta = all_meta[batch_start:batch_start + args.batch_size]
        
        # Render prompts
        rendered = []
        for m in batch_meta:
            msgs = []
            if m["system"]:
                msgs.append({"role": "system", "content": m["system"]})
            msgs.append({"role": "user", "content": m["instruction"]})
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            rendered.append(text)

        t1 = time.time()
        outputs = llm.generate(rendered, sampling)
        elapsed = time.time() - t1
        print(f"[gen] batch {batch_start}-{batch_start+len(batch_meta)}: {len(outputs)} in {elapsed:.1f}s ({len(outputs)/elapsed:.1f} req/s)")

        # Write output
        with open(out_path, 'a') as f:
            for out, m in zip(outputs, batch_meta):
                response = out.outputs[0].text if out.outputs else ""
                rec = {
                    "prompt_id": m["prompt_id"],
                    "language": m["language"],
                    "source": m["source"],
                    "n_idx": m["n_idx"],
                    "response": response,
                    "response_len": len(response),
                    "model": args.model,
                    "temp": args.temp,
                    "top_p": args.top_p,
                    "presence_penalty": args.presence_penalty,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                total_written += 1
                all_lens.append(len(response))

    print(f"\n[done] wrote {total_written} -> {out_path}")
    if all_lens:
        arr = np.array(all_lens)
        print(f"  response len: mean={arr.mean():.0f} std={arr.std():.0f} min={arr.min()} max={arr.max()}")

    # Per-language stats
    by_lang = defaultdict(list)
    for m, l in zip(all_meta, all_lens):
        by_lang[m['language']].append(l)
    print("\nPer-language response lengths:")
    for lang in sorted(by_lang.keys()):
        arr = np.array(by_lang[lang])
        print(f"  {lang}: N={len(arr)} mean={arr.mean():.0f}")

if __name__ == "__main__":
    main()
