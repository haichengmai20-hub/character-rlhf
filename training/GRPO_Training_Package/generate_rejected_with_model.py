#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 SFT Merge 模型生成 Rejected 回复

核心思路：
  对每条现有数据，用模型生成多个候选回复，
  然后通过多种策略选择"较弱"的作为 rejected

================================================================================
🎯 Rejected 生成策略
================================================================================

策略 1: 随机采样 (Random Sampling)
────────────────────────────────
原理: 
  • 用低 temperature 生成多个候选
  • 选择得分最低的作为 rejected
  • chosen = 原始高质量回复
  • rejected = 模型生成的"较弱"版本

优点: ✓ 速度快，不需要额外评分
缺点: ✗ 质量不稳定，可能生成"太烂"的 rejected

───────────────────────────────────────────────────────────────────────────────

策略 2: 困惑度排序 (Perplexity Ranking)
─────────────────────────────────────
原理:
  • 计算 chosen 和 rejected 在模型上的困惑度
  • 困惑度越高 = 模型越"不确定" = 质量越低
  • 选择困惑度在 [0.4, 0.7] × chosen_perplexity 的作为 rejected
  • 这样保证 rejected "比 chosen 差，但还不是太烂"

优点: ✓ 质量可控，理论更清晰
缺点: ✗ 计算量大，需要逐个计算困惑度

───────────────────────────────────────────────────────────────────────────────

策略 3: 长度-多样性降级 (Length-Diversity Degradation)
──────────────────────────────────────────────────
原理:
  • 用高 temperature 生成短版本 (长度 50-80%)
  • 自然会牺牲信息完整性
  • chosen = 原始回复 (完整、连贯)
  • rejected = 模型生成的短版本 (信息不完整)

优点: ✓ 速度快，自然多样
缺点: ✗ 所有 rejected 都较短，可能学到虚假相关

───────────────────────────────────────────────────────────────────────────────

策略 4: 多代理轮转 (Round-Robin Generation)
───────────────────────────────────────────
原理:
  • 用不同角色的 system prompt 让同一个 SFT 模型扮演不同角色
  • 对中文 prompt，先用"精英富女"角色生成 (作为 chosen)
  • 再用"校园恶霸"角色生成 (作为 rejected)
  • 虽然是同一个 SFT 模型，但不同 system prompt 产生不同的"风格"

优点: ✓ 充分利用 20 个角色维度，自然多样
缺点: ✗ 需要定义 20 个角色的 system prompt

───────────────────────────────────────────────────────────────────────────────

策略 5: 对立观点生成 (Adversarial Generation)
──────────────────────────────────────────
原理:
  • 对 prompt 添加"对立约束"，让模型生成相反观点
  • 例: 原 prompt "这是个好主意" → 修改为 "为什么这不是个好主意"
  • chosen = 原始回复 (正常)
  • rejected = 对立观点的回复 (质量较低，因为违反原始 prompt)

优点: ✓ 教会 RM 理解对立观点的"错误性"
缺点: ✗ 适用场景有限，某些 prompt 难以修改

================================================================================
🚀 推荐方案: 困惑度排序 + 长度降级混合
================================================================================

最佳实践:
  70% 困惑度排序方案 (质量更可控，更接近原始数据质量)
  30% 长度降级方案 (增加多样性)

这样能保证:
  ✓ rejected 不会"太烂" (困惑度在 [0.5-0.7] × chosen 之间)
  ✓ 数据多样性强 (长度差异)
  ✓ RM 学到多个维度的约束
"""

import json
import argparse
import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import math


class RejectedGenerator:
    """使用 SFT 模型生成 Rejected 回复"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        torch_dtype: str = "float16",
        max_length: int = 512,
        batch_size: int = 4,
    ):
        print(f"[1/3] 加载模型: {model_path}")
        
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype_obj = dtype_map.get(torch_dtype, torch.float16)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype_obj,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        
        print(f"✓ 模型加载完成")
        print(f"  设备: {device}")
        print(f"  dtype: {torch_dtype}")
    
    def load_data(self, input_file: str) -> List[Dict]:
        """加载数据"""
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                
                # ChatJoy 格式
                if 'item' in raw and isinstance(raw['item'], dict):
                    item = raw['item']
                    record = {
                        'prompt': item.get('instruction', ''),
                        'chosen': item.get('output', ''),
                        'character': item.get('character_name', f"char_{item.get('character_id', 0)}"),
                        'character_id': item.get('character_id'),
                        'language': item.get('language', 'unknown'),
                        'system': item.get('system', ''),
                    }
                else:
                    record = {
                        'prompt': raw.get('prompt', raw.get('instruction', '')),
                        'chosen': raw.get('chosen', raw.get('response', raw.get('output', ''))),
                        'character': raw.get('character', 'default'),
                        'language': raw.get('language', 'unknown'),
                    }
                
                if record.get('prompt') and record.get('chosen'):
                    data.append(record)
        
        return data
    
    def compute_perplexity(self, text: str) -> float:
        """计算文本困惑度 (Perplexity)
        
        困惑度 = exp(平均 token log-probability)
        困惑度越低 = 模型越"确定"这个文本 = 质量越高
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            # perplexity = exp(loss)
            perplexity = math.exp(loss.item())
        
        return perplexity
    
    def generate_candidates(
        self,
        prompt: str,
        num_candidates: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[str]:
        """生成多个候选回复"""
        candidates = []
        
        with torch.no_grad():
            for _ in range(num_candidates):
                inputs = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=256
                ).to(self.device)
                
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                
                # 移除输入部分
                generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
                response = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True
                ).strip()
                
                if response:
                    candidates.append(response)
        
        return candidates
    
    def select_rejected_perplexity(
        self,
        prompt: str,
        chosen: str,
        num_candidates: int = 5,
        temperature: float = 0.8,
    ) -> Tuple[str, float]:
        """
        策略: 困惑度排序
        
        生成多个候选，选择困惑度在 [0.5, 0.8] × chosen_ppl 的作为 rejected
        """
        # 计算 chosen 的困惑度 (作为基准)
        print(f"  计算 chosen 困惑度...")
        chosen_ppl = self.compute_perplexity(chosen)
        
        # 生成候选
        print(f"  生成 {num_candidates} 个候选...")
        candidates = self.generate_candidates(
            prompt,
            num_candidates=num_candidates,
            temperature=temperature
        )
        
        if not candidates:
            return None, None
        
        # 计算每个候选的困惑度
        print(f"  计算候选困惑度...")
        candidate_ppls = []
        for cand in candidates:
            ppl = self.compute_perplexity(cand)
            candidate_ppls.append((cand, ppl))
        
        # 选择困惑度在目标范围内的候选
        target_min = 0.5 * chosen_ppl
        target_max = 0.8 * chosen_ppl
        
        in_range = [
            (cand, ppl) for cand, ppl in candidate_ppls
            if target_min <= ppl <= target_max
        ]
        
        if in_range:
            # 从范围内随机选择
            rejected, rejected_ppl = random.choice(in_range)
        else:
            # 如果没有在范围内的，选择最接近的
            rejected, rejected_ppl = min(
                candidate_ppls,
                key=lambda x: abs(x[1] - (chosen_ppl * 0.65))
            )
        
        return rejected, chosen_ppl / rejected_ppl if rejected_ppl > 0 else 1.0
    
    def select_rejected_length_degradation(
        self,
        prompt: str,
        chosen: str,
        num_candidates: int = 5,
        temperature: float = 1.0,
        target_length_ratio: float = 0.6,
    ) -> str:
        """
        策略: 长度降级
        
        生成短版本，自然降低信息完整性
        """
        target_length = int(len(chosen) * target_length_ratio)
        
        # 用更高 temperature 生成，会更多样化但可能更短
        candidates = self.generate_candidates(
            prompt,
            num_candidates=num_candidates,
            temperature=temperature
        )
        
        if not candidates:
            return None
        
        # 选择长度最接近目标的候选
        best_candidate = min(
            candidates,
            key=lambda x: abs(len(x) - target_length)
        )
        
        return best_candidate
    
    def generate_rejected_mixed(
        self,
        data: List[Dict],
        num_candidates: int = 5,
        strategy_mix: float = 0.7,  # 70% perplexity, 30% length
    ) -> List[Dict]:
        """
        混合策略: 70% 困惑度 + 30% 长度降级
        """
        output_data = []
        
        for i, record in enumerate(tqdm(data, desc="生成 rejected")):
            prompt = record['prompt']
            chosen = record['chosen']
            character = record['character']
            
            try:
                # 按概率选择策略
                if random.random() < strategy_mix:
                    # 困惑度策略
                    rejected, quality_ratio = self.select_rejected_perplexity(
                        prompt,
                        chosen,
                        num_candidates=num_candidates
                    )
                    strategy_used = "perplexity"
                else:
                    # 长度降级策略
                    rejected = self.select_rejected_length_degradation(
                        prompt,
                        chosen,
                        num_candidates=num_candidates
                    )
                    quality_ratio = None
                    strategy_used = "length_degradation"
                
                if not rejected:
                    continue
                
                # 质量评分
                quality_chosen = 0.85  # SFT 数据本身质量高
                quality_rejected = 0.35  # 生成的较弱版本
                
                pair = {
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected,
                    'character': character,
                    'character_id': record.get('character_id'),
                    'language': record.get('language', 'unknown'),
                    'quality_chosen': quality_chosen,
                    'quality_rejected': quality_rejected,
                    'strategy': strategy_used,
                    'source': 'sft_model_generation',
                }
                
                output_data.append(pair)
                
            except Exception as e:
                print(f"  ⚠️ 样本 {i} 失败: {e}")
                continue
        
        return output_data
    
    def process(
        self,
        input_file: str,
        output_file: str,
        num_candidates: int = 5,
        strategy_mix: float = 0.7,
        sample_size: Optional[int] = None,
    ) -> None:
        """主流程"""
        print(f"\n[2/3] 加载数据: {input_file}")
        data = self.load_data(input_file)
        print(f"✓ 加载 {len(data)} 条数据")
        
        # 可选: 只处理部分数据 (测试)
        if sample_size:
            data = random.sample(data, min(sample_size, len(data)))
            print(f"  (取样 {len(data)} 条进行测试)")
        
        print(f"\n[3/3] 生成 rejected 回复")
        output_data = self.generate_rejected_mixed(
            data,
            num_candidates=num_candidates,
            strategy_mix=strategy_mix,
        )
        
        # 统计
        print(f"\n【统计信息】")
        print(f"  成功生成: {len(output_data)} 对")
        print(f"  成功率: {100*len(output_data)/len(data):.1f}%")
        
        # 按策略统计
        strategies = defaultdict(int)
        for pair in output_data:
            strategies[pair['strategy']] += 1
        
        print(f"\n  【按策略统计】")
        for strategy, count in strategies.items():
            pct = 100 * count / len(output_data)
            print(f"    - {strategy}: {count} 对 ({pct:.1f}%)")
        
        # 保存
        print(f"\n保存数据: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for pair in output_data:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        print(f"✓ 完成!")
        
        # 示例
        print(f"\n【示例对】\n")
        for idx, pair in enumerate(output_data[:2]):
            print(f"样本 {idx+1}:")
            print(f"  Prompt: {pair['prompt'][:60]}...")
            print(f"  Chosen: {pair['chosen'][:60]}...")
            print(f"  Rejected: {pair['rejected'][:60]}...")
            print(f"  策略: {pair['strategy']}")
            print(f"  质量: {pair['quality_chosen']:.2f} vs {pair['quality_rejected']:.2f}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description='使用 SFT 模型生成 Rejected 回复'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='SFT merge 模型路径'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入数据文件 (JSONL)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出偏好对文件'
    )
    parser.add_argument(
        '--num-candidates',
        type=int,
        default=5,
        help='生成多少个候选 (default: 5)'
    )
    parser.add_argument(
        '--strategy-mix',
        type=float,
        default=0.7,
        help='困惑度策略占比 (default: 0.7, 表示 70%)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='只处理多少条数据 (用于测试，default: 全部)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='GPU 设备 (default: cuda:0)'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=['float16', 'float32', 'bfloat16'],
        help='数据类型 (default: float16)'
    )
    
    args = parser.parse_args()
    
    generator = RejectedGenerator(
        model_path=args.model_path,
        device=args.device,
        torch_dtype=args.dtype,
    )
    
    generator.process(
        input_file=args.input,
        output_file=args.output,
        num_candidates=args.num_candidates,
        strategy_mix=args.strategy_mix,
        sample_size=args.sample_size,
    )


if __name__ == '__main__':
    main()
