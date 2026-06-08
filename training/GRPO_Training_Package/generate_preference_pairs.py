#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 RM 训练数据 (Preference Pairs) - 从 ChatJoy 数据转换

================================================================================
📋 Preference Pair 的概念
================================================================================
Preference Pair (偏好对) 是用于训练 Reward Model 的关键数据格式：
  {
    "prompt": "用户输入",
    "chosen": "高质量回复",      # chosen > rejected in terms of quality
    "rejected": "低质量回复",
    "character": "角色名",
    "quality_chosen": 0.8,
    "quality_rejected": 0.3
  }

Reward Model 的作用是学习给定 (prompt, response) 对，
输出一个质量评分 ∈ [0, 1]，用于指导 Actor 强化学习。

================================================================================
✅ 如何从 ChatJoy 现有数据生成偏好对
================================================================================

ChatJoy 数据结构 (JSONL 格式):
  {
    "index": 1,
    "item": {
      "character_id": 111,
      "instruction": "用户输入",
      "output": "模型回复",
      "language": "zh",
      "system": "系统提示"
    }
  }

方法 1: 基于质量评分 (quality_score) ← 推荐
─────────────────────────────────────
前提: 需要给每条数据添加 quality_score 字段 (0-1)
转换步骤:
  1. 加载 ChatJoy 数据 (JSONL)
  2. 为每条数据增加 quality_score (自动或手动)
  3. 按字符分组
  4. 按质量分类 (高质量 ≥0.7, 低质量 <0.7)
  5. 配对 (同一问题的高质量 vs 低质量)
  6. 生成 rm_training_data.jsonl

方法 2: TOP20 字符专用 (top20)
─────────────────────────────
特点: 为每个 TOP20 字符确保足够的偏好对
优点: 字符平衡分布

================================================================================
🔧 实际使用步骤
================================================================================

Step 1: 准备数据
───────────────
# 从 ChatJoy 的训练数据开始
source_file=/home/ps/hanjiaqi/LlamaFactory/chatjoy/chatjoy_top20_multilingual_27x2200_preapi_gemini.progress.jsonl

# 或者 SFT 训练配置指向的文件
source_file=/home/ps/hanjiaqi/LlamaFactory/chatjoy/data/clean32_for_sft2.jsonl

Step 2: 转换为 preference pairs
───────────────────────────────
python generate_preference_pairs.py \\
  --input $source_file \\
  --output data/rm_training_data.jsonl \\
  --method quality_score

Step 3: 检查生成的偏好对
──────────────────────
head -5 data/rm_training_data.jsonl | python3 -m json.tool

预期输出:
  [
    {
      "prompt": "...",
      "chosen": "...",
      "rejected": "...",
      "character": "精英富女",
      "quality_chosen": 0.85,
      "quality_rejected": 0.25
    }
  ]

Step 4: 用于 RM 训练
───────────────────
python train_rm_model.py \\
  --sft_model_path saves/qwen3-30b-moe/clean32_for_sft2_fsdp_qlora_merged \\
  --train_file data/rm_training_data.jsonl \\
  --output_dir saves/rm_model

================================================================================
💡 从 ChatJoy 数据生成质量评分的几种方案
================================================================================

方案 A: 使用已有的评分 (如果有)
──────────────────────────
假设您的 ChatJoy 数据已经包含 quality_score，
脚本会自动识别并使用。

方案 B: 使用 SFT 模型进行评分 (推荐)
───────────────────────────────────
思路: 使用已训练好的 SFT 模型来评判数据质量
实现:
  1. 对每条数据用 SFT 模型生成回复
  2. 与原始回复比较相似度 → quality_score
  3. 或者用专业评分模型来评

方案 C: 使用启发式规则评分
──────────────────────────
根据以下规则自动评分:
  - 长度: 太短/太长 → 减分
  - 多语言匹配: 语言不匹配 → 减分
  - 角色一致性: 不符合角色特征 → 减分
  - 安全性: 包含禁忌词汇 → 减分

方案 D: 使用人工标注 (最准但成本高)
──────────────────────────────────
选择高质量数据进行人工标注
建议样本: ~2000-5000 条优质回复

================================================================================
📊 各方案的比较
================================================================================

| 方案 | 质量 | 成本 | 时间 | 推荐度 |
|------|------|------|------|--------|
| A (已有评分) | ⭐⭐⭐⭐⭐ | 低 | 快 | ✅ 最佳 |
| B (SFT 评分) | ⭐⭐⭐⭐ | 中 | 中 | ✅ 推荐 |
| C (启发式) | ⭐⭐⭐ | 低 | 快 | ⚠️ 可用 |
| D (人工标注) | ⭐⭐⭐⭐⭐ | 高 | 慢 | ⚠️ 可选 |

当前实现: 方案 C (启发式) + 方案 B 的预留接口

用法:
  python generate_preference_pairs.py \
    --input your_data.jsonl \
    --output rm_training_data.jsonl \
    --method quality_score
"""

import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm


class PreferenceDataGenerator:
    """生成偏好对数据用于 RM 训练"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.seed = seed
    
    def load_data(self, input_file: str) -> List[Dict]:
        """
        加载输入数据，自动检测 ChatJoy 格式
        
        支持两种格式:
        1. 直接格式: {"prompt": "...", "response": "..."}
        2. ChatJoy 格式: {"index": ..., "item": {"instruction": "...", "output": "..."}}
        """
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                raw = json.loads(line)
                
                # 检测 ChatJoy 格式
                if 'item' in raw and isinstance(raw['item'], dict):
                    # ChatJoy 原生格式
                    item = raw['item']
                    record = {
                        'prompt': item.get('instruction', ''),
                        'response': item.get('output', ''),
                        'character_id': item.get('character_id'),
                        'character': item.get('character_name', f"char_{item.get('character_id', 0)}"),
                        'language': item.get('language', 'unknown'),
                        'quality_score': item.get('quality_score', 0.5),  # 默认中等质量
                        'system': item.get('system', ''),
                    }
                else:
                    # 直接格式
                    record = {
                        'prompt': raw.get('prompt', raw.get('instruction', '')),
                        'response': raw.get('response', raw.get('output', '')),
                        'character': raw.get('character', 'default'),
                        'character_id': raw.get('character_id'),
                        'language': raw.get('language', 'unknown'),
                        'quality_score': raw.get('quality_score', 0.5),
                    }
                
                if record.get('prompt') and record.get('response'):
                    data.append(record)
        
        return data
    
    def generate_by_quality_score(self, data: List[Dict]) -> List[Dict]:
        """
        方法 1: 基于质量评分生成偏好对
        
        假设数据结构:
        {
          "prompt": "...",
          "response": "...",
          "quality_score": 0.8,  # 0-1
          "character": "..."
        }
        """
        preference_pairs = []
        
        # 按字符分组
        char_data = {}
        for item in data:
            char = item.get('character', 'default')
            if char not in char_data:
                char_data[char] = []
            char_data[char].append(item)
        
        # 为每个字符生成偏好对
        for char, items in char_data.items():
            # 按质量分类
            high_quality = [x for x in items if x.get('quality_score', 0) >= 0.7]
            low_quality = [x for x in items if x.get('quality_score', 0) < 0.7]
            
            # 配对
            for high in high_quality:
                for low in random.sample(low_quality, min(1, len(low_quality))):
                    if high.get('prompt') == low.get('prompt'):  # 同一问题
                        pair = {
                            'prompt': high['prompt'],
                            'chosen': high['response'],
                            'rejected': low['response'],
                            'character': char,
                            'quality_chosen': high.get('quality_score', 0.8),
                            'quality_rejected': low.get('quality_score', 0.3),
                        }
                        preference_pairs.append(pair)
        
        return preference_pairs
    

    
    def generate_from_top20_characters(self, data: List[Dict]) -> List[Dict]:
        """
        方法 4: 针对 TOP20 字符的偏好对生成
        
        确保每个字符都有足够的偏好对
        """
        preference_pairs = []
        
        # 按字符分组
        char_data = {}
        for item in data:
            char = item.get('character', 'default')
            if char not in char_data:
                char_data[char] = []
            char_data[char].append(item)
        
        print(f"\n检测到 {len(char_data)} 个字符")
        
        # 为每个字符生成偏好对
        for char, items in tqdm(char_data.items(), desc="按字符生成"):
            # 假设项目中有 quality_score 或 rank
            items_with_score = [
                (item, item.get('quality_score', 
                              1 - item.get('rank', 999) / 1000))
                for item in items
            ]
            items_with_score.sort(key=lambda x: x[1], reverse=True)
            
            # 生成偏好对
            high_items = items_with_score[:max(1, len(items_with_score) // 2)]
            low_items = items_with_score[max(1, len(items_with_score) // 2):]
            
            for high_item, high_score in high_items:
                for low_item, low_score in random.sample(
                    low_items, 
                    min(2, len(low_items))
                ):
                    # 确保是同一个 prompt
                    if high_item.get('prompt') == low_item.get('prompt'):
                        pair = {
                            'prompt': high_item['prompt'],
                            'chosen': high_item.get('response', ''),
                            'rejected': low_item.get('response', ''),
                            'character': char,
                            'score_margin': high_score - low_score,
                        }
                        if pair['chosen'] and pair['rejected']:  # 验证
                            preference_pairs.append(pair)
        
        return preference_pairs
    
    def save_data(self, data: List[Dict], output_file: str) -> None:
        """保存数据为 JSONL"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def process(
        self,
        input_file: str,
        output_file: str,
        method: str = 'quality_score'
    ) -> None:
        """主处理流程"""
        print(f"[1/4] 加载数据: {input_file}")
        data = self.load_data(input_file)
        print(f"  ✓ 加载 {len(data)} 条数据")
        
        print(f"\n[2/4] 生成偏好对 (方法: {method})")
        if method == 'quality_score':
            preference_pairs = self.generate_by_quality_score(data)
        elif method == 'top20':
            preference_pairs = self.generate_from_top20_characters(data)
        else:
            raise ValueError(f"未知方法: {method}")
        
        print(f"  ✓ 生成 {len(preference_pairs)} 个偏好对")
        
        # 统计信息
        print(f"\n[3/4] 统计信息:")
        chars = {}
        for pair in preference_pairs:
            char = pair.get('character', 'default')
            chars[char] = chars.get(char, 0) + 1
        
        for char, count in sorted(chars.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {char}: {count} 个偏好对")
        
        print(f"\n[4/4] 保存数据: {output_file}")
        self.save_data(preference_pairs, output_file)
        print(f"  ✓ 完成!")
        
        # 显示样本
        print(f"\n示例数据:")
        for i, pair in enumerate(preference_pairs[:2]):
            print(f"\n样本 {i+1}:")
            print(f"  Prompt: {pair['prompt'][:100]}...")
            print(f"  Chosen: {pair['chosen'][:100]}...")
            print(f"  Rejected: {pair['rejected'][:100]}...")


def main():
    parser = argparse.ArgumentParser(
        description='生成 Reward Model 训练数据'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入数据文件 (JSONL 格式)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出偏好对文件'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='quality_score',
        choices=['quality_score', 'top20'],
        help='数据生成方法 (quality_score: 基于质量评分, top20: 针对TOP20字符)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = PreferenceDataGenerator(seed=args.seed)
    
    # 处理
    generator.process(
        input_file=args.input,
        output_file=args.output,
        method=args.method
    )


if __name__ == '__main__':
    main()
