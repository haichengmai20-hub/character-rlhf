#!/usr/bin/env python3
"""run_sweep.py
串行执行全部实验:
  1. step0 × temp {0.8, 0.85, 0.9} — 温度甜点扫描
  2. step100 × temp 1.0 — 热态防御检验
  3. 统一 RM 打分
  4. 汇总分析

用法:
  /home/ubuntu/miniconda3/envs/gemma4-rl/bin/python3 run_sweep.py
"""
import subprocess, sys, os, json, time
import numpy as np

PYTHON = "/home/ubuntu/miniconda3/envs/gemma4-rl/bin/python3"
BASE = "/root/outputs/v67_step0_vs_step100"
STEP0_MODEL = "/root/sft_rich_cjk_v67_repair_merged_full"
STEP100_MODEL = None  # 需要确认

# 先找 step100 模型路径
def find_step100_model():
    """从 step100_probe.jsonl 的 model 字段推断，或从 saves 目录找"""
    # 常见路径
    candidates = [
        "/root/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/rl_v6.5_step425_merged",
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.exists(os.path.join(p, "config.json")):
            return p
    return None

RM_MODEL = "/root/hanjiaqi/LlamaFactory/chatjoy/saves/gemma4-26b-a4b/chatjoy_rm_v6.6_fg_7gpu_merged_vllm"

# 攻防 RM (attack_rm) 路径 — 如果有的话
ATTACK_RM = None  # TODO: 确认路径


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"[RUN] {desc}")
    print(f"  cmd: {cmd}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False, timeout=600)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[FAIL] {desc} — exit code {result.returncode} after {elapsed:.1f}s")
        return False
    print(f"[OK] {desc} — {elapsed:.1f}s")
    return True


def score_with_rm(probe_path, out_path, desc):
    """用 RM 给 probe jsonl 打分"""
    script = os.path.join(BASE, "score_probe_rm.py")
    # score_probe_rm.py 读取 data-dir 下的 base/step0/step100
    # 我们需要一个能指定单文件的打分方式，直接内联
    cmd = f"""{PYTHON} -c "
import json, os, time, numpy as np
from vllm import LLM
from transformers import AutoTokenizer

probe_path = '{probe_path}'
out_path = '{out_path}'
rm_path = '{RM_MODEL}'

tok = AutoTokenizer.from_pretrained(rm_path, trust_remote_code=True)
records = []
with open(probe_path) as f:
    for line in f:
        records.append(json.loads(line))
print(f'[score] {{len(records)}} records')

prompts = []
for rec in records:
    msgs = []
    if rec.get('system'): msgs.append({{'role':'system','content':rec['system']}})
    if rec.get('user'): msgs.append({{'role':'user','content':rec['user']}})
    if rec.get('response'): msgs.append({{'role':'assistant','content':rec['response']}})
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    prompts.append(rendered)

t0 = time.time()
llm = LLM(model=rm_path, runner='pooling', tensor_parallel_size=4,
          gpu_memory_utilization=0.85, max_model_len=8192,
          limit_mm_per_prompt={{'image':0,'video':0}},
          trust_remote_code=True, enforce_eager=True)
print(f'[load] RM in {{time.time()-t0:.1f}}s')

t1 = time.time()
outs = llm.classify(prompts)
print(f'[score] scored {{len(prompts)}} in {{time.time()-t1:.1f}}s')

scores = []
with open(out_path, 'w') as f:
    for rec, ro in zip(records, outs):
        o = ro.outputs
        probs = getattr(o, 'probs', None)
        s = float(list(probs)[0]) if probs is not None else 0.0
        scores.append(s)
        out_rec = {{
            'group': rec.get('model',''),
            'char_id': rec.get('char_id',''),
            'case_id': rec.get('case_id',''),
            'lang': rec.get('lang',''),
            'rm_score': s,
        }}
        f.write(json.dumps(out_rec, ensure_ascii=False) + chr(10))

arr = np.array(scores)
print(f'[done] mean={{arr.mean():.4f}} std={{arr.std():.4f}} min={{arr.min():.4f}} max={{arr.max():.4f}} n={{len(arr)}}')
"
"""
    return run(cmd, desc)


def main():
    # 实验 1: step0 温度甜点扫描
    step0_probe = os.path.join(BASE, "step0_probe.jsonl")
    temps = [0.8, 0.85, 0.9]
    
    for temp in temps:
        out = os.path.join(BASE, f"step0_t{temp}.jsonl")
        cmd = (f"{PYTHON} {BASE}/resample_probe.py "
               f"--model {STEP0_MODEL} "
               f"--probe {step0_probe} "
               f"--out {out} "
               f"--temp {temp} --top-p 0.95 --presence-penalty 0.4")
        if not run(cmd, f"step0 temp={temp}"):
            print(f"[SKIP] step0 temp={temp} failed, skipping RM score")

    # 实验 2: step100 热态采样
    step100_probe = os.path.join(BASE, "step100_probe.jsonl")
    step100_model = find_step100_model()
    
    if step100_model:
        print(f"\n[INFO] step100 model: {step100_model}")
        step100_hot_out = os.path.join(BASE, "step100_t1.0.jsonl")
        cmd = (f"{PYTHON} {BASE}/resample_probe.py "
               f"--model {step100_model} "
               f"--probe {step100_probe} "
               f"--out {step100_hot_out} "
               f"--temp 1.0 --top-p 0.95 --presence-penalty 0.4")
        if not run(cmd, "step100 temp=1.0"):
            print("[SKIP] step100 hot sampling failed")
    else:
        print("\n[WARN] step100 model not found! Checking probe for model field...")
        # 从 step100_probe 读 model 字段看能不能推断
        with open(step100_probe) as f:
            first = json.loads(f.readline())
            print(f"  step100 probe model field: {first.get('model')}")
        print("[SKIP] step100 hot sampling — need model path")

    # 实验 3: 统一 RM 打分
    print(f"\n{'='*60}")
    print("[PHASE] RM Scoring all new probes")
    print(f"{'='*60}")
    
    all_probes = []
    for temp in temps:
        probe = os.path.join(BASE, f"step0_t{temp}.jsonl")
        score_out = os.path.join(BASE, f"step0_t{temp}_rm.jsonl")
        if os.path.exists(probe):
            all_probes.append((probe, score_out, f"step0_t{temp}"))
    
    step100_hot = os.path.join(BASE, "step100_t1.0.jsonl")
    if os.path.exists(step100_hot):
        all_probes.append((step100_hot, os.path.join(BASE, "step100_t1.0_rm.jsonl"), "step100_t1.0"))
    
    for probe, score_out, desc in all_probes:
        score_with_rm(probe, score_out, f"RM score: {desc}")

    print(f"\n{'='*60}")
    print("[DONE] All experiments complete. Run analyze_sweep.py for summary.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
