#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级 Preference Pair 生成 - 多角色多语言方案
针对高质量数据，通过语言/角色错配和多样性生成 rejected 

================================================================================
🎯 核心问题与解决方案
================================================================================

问题:
  - 您的 ChatJoy 数据都是高质量 (已过 SFT 选择)
  - 无法直接提取 "低质量" 回复作为 rejected
  - 需要保留 20 个角色、多语言的多样性

解决方案: 基于错配和质量差异生成 rejected

================================================================================
💡 Rejected 生成策略 (5 种)
================================================================================

策略 1: 角色错配 (Role Mismatch)
✓ 同一 prompt，多个角色都有回复
✓ chosen: 原始角色的回复 (质量: 0.9)
✓ rejected: 其他角色的回复 (质量: 0.4-0.6)
✓ 优点: 自然且有意义，保留多样性
✓ 多样性: 20 个角色 → 每个 prompt 有 19+ 个 rejected 候选

  例子:
  prompt: "你好，今天天气怎么样?"
  chosen: 精英富女说的回复 (质量: 0.9)
  rejected: 校园恶霸说的回复 (质量: 0.5)  ← 角色不匹

策略 2: 语言错配 (Language Mismatch)
✓ 同一含义的 prompt，多语言有回复
✓ chosen: 语言匹配的回复 (质量: 0.9)
✓ rejected: 语言不匹配的回复 (质量: 0.3)
✓ 优点: 教会 RM 语言一致性重要性
✓ 多样性: 27 种语言 → 每个 prompt 有 26+ 个 rejected 候选

  例子:
  prompt_zh: "你好" → 中文回复 (质量: 0.9)
  Chinese response: "很高兴认识你" 
  rejected: 英文回复 (质量: 0.3) ← 语言错配

策略 3: 长度和信息量降低
✓ 同一 response 的截断版本
✓ chosen: 完整回复 (质量: 0.9)
✓ rejected: 截断到 50% 的版本 (质量: 0.4)
✓ 优点: 简单快速，无需外部数据
✓ 多样性: 每个 response 可生成多个截断版本

  例子:
  chosen: "午好！今天天气很好，阳光明媚，建议出去走走。😊"
  rejected: "午好！今天天气" ← 信息不完整

策略 4: 角色+语言混合错配
✓ 组合上述两种错配
✓ chosen: (角色匹配 + 语言匹配)，质量: 0.95
✓ rejected: (角色错配 OR 语言错配)，质量: 0.2-0.5
✓ 优点: 最强的对比，RM 学到多个维度
✓ 难度: 高，但数据丰富时效果最好

  例子:
  prompt_zh: "你好，我是精英富女"
  chosen: 精英富女的中文回复 (质量: 0.95)
  rejected: 校园恶霸的英文回复 (质量: 0.25) ← 角色+语言都错

策略 5: 质量层级差异
✓ 对性质相同但细节不同的多个回复排序
✓ chosen: 最详细、最符合角色的版本 (质量: 0.9)
✓ rejected: 较生硬、较简短的版本 (质量: 0.5)
✓ 优点: 教会 RM 微妙的质量差异
✓ 多样性: 需要多个版本数据

================================================================================
🔧 推荐实现方案 (适合您的情况)
================================================================================

【方案 A】: 角色错配优先 (推荐度: ⭐⭐⭐⭐⭐)
─────────────────────────────────────
前提: ChatJoy 有 20 个角色的数据
优点: 最自然、最有意义、多样性最强
实现:
  1. 按 prompt 聚类 (相同含义的 prompt 组在一起)
  2. 每个 prompt 的不同角色回复 = {chosen 1 + rejected 19}
  3. 形成 19 个偏好对
  4. 验证质量: chosen 是原始角色，rejected 是其他角色

示例:
```
prompt: "你好" × 20 个角色回复
├─ 精英富女版本: "很高兴认识你，请问有什么我可以帮你的吗？"
├─ 校园恶霸版本: "你是谁？" 
├─ 撒娇女儿版本: "⭐️ 你好呀 ~"
├─ ... (其他 17 个)

转化为:
Pair1: chosen=精英富女, rejected=校园恶霸 (quality: 0.8 vs 0.3)
Pair2: chosen=精英富女, rejected=撒娇女儿 (quality: 0.8 vs 0.5)
...
Pair19: chosen=精英富女, rejected=其他角色
```

【方案 B】: 语言错配 (推荐度: ⭐⭐⭐⭐)
────────────────────────────────
前提: 同一个 prompt 有多语言版本
优点: 教会 RM 语言匹配
实现:
  1. 聚类相同语义的 prompts (不同语言版本)
  2. 每个语言的回复 = {chosen + rejected(其他语言)}
  3. 质量分差异: 语言匹配 0.9 vs 不匹配 0.3

示例:
```
prompt_zh: "你好" → 中文回复: "很高兴认识你"
prompt_en: "Hello" → 英文回复: "Nice to meet you"
prompt_ja: "こんにちは" → 日文回复: "お会いできて嬉しいです"

Pair1: prompt=你好, chosen=中文回复, rejected=英文回复 (0.9 vs 0.3)
Pair2: prompt=Hello, chosen=英文回复, rejected=日文回复 (0.9 vs 0.3)
```

【方案 C】: 长度/质量降级 (推荐度: ⭐⭐⭐)
─────────────────────────────────────
前提: 仅需要每个 response 一次
优点: 快速，不需要额外数据
实现:
  1. 对每个 response，生成 2-3 个截断/简化版本
  2. 质量等级: 完整 > 80% > 60% > 40%
  3. chosen = 完整 (0.9), rejected = 截断 (0.4-0.6)

示例:
```
original: "午好！今天天气很好，阳光明媚，非常适合出去走走。😊"
v80%:     "午好！今天天气很好，阳光明媚，非常适合出去走走。"
v60%:     "午好！今天天气很好，阳光明媚，适合出去。"
v40%:     "午好！天气很好。"

Pair1: chosen=original, rejected=v80% (0.85 vs 0.70)
Pair2: chosen=original, rejected=v60% (0.85 vs 0.50)
```

================================================================================
📊 方案对比
================================================================================

| 方案 | 多样性 | 工作量 | 自然度 | 推荐度 |
|------|--------|-------|--------|--------|
| A: 角色错配 | ⭐⭐⭐⭐⭐ | 中 | ⭐⭐⭐⭐⭐ | ✅ 最佳 |
| B: 语言错配 | ⭐⭐⭐⭐ | 中 | ⭐⭐⭐⭐ | ✅ 优秀 |
| C: 长度降级 | ⭐⭐⭐ | 低 | ⭐⭐⭐ | ⚠️ 可用 |
| A+B 混合 | ⭐⭐⭐⭐⭐ | 高 | ⭐⭐⭐⭐⭐ | 🚀 最强 |

================================================================================
🚀 立即使用
================================================================================

使用方法:
  python generate_preference_pairs_advanced.py \\
    --input chatjoy_data.jsonl \\
    --output rm_training_data.jsonl \\
    --method multi_role    # 改用角色错配策略

支持的方法:
  --method multi_role      # 角色错配 (推荐)
  --method multi_language  # 语言错配
  --method multi_length    # 长度层级
  --method multi_combined  # 角色+语言混合
"""

import json
import argparse
import random
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm


class AdvancedPreferenceDataGenerator:
    """多角色多语言偏好对生成"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.seed = seed
    
    def load_data(self, input_file: str) -> List[Dict]:
        """加载 ChatJoy 数据，识别角色和语言"""
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
                        'response': item.get('output', ''),
                        'character_id': item.get('character_id'),
                        'character': item.get('character_name', f"char_{item.get('character_id', 0)}"),
                        'language': item.get('language', 'unknown'),
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
                    }
                
                if record.get('prompt') and record.get('response'):
                    data.append(record)
        
        return data
    
    def generate_by_multi_role(self, data: List[Dict]) -> List[Dict]:
        """
        【方案 A】角色错配: 同一 prompt，不同角色竞争
        
        核心思路:
        1. 按 prompt 聚类
        2. 同一 prompt 的不同角色 = chosen + rejected
        3. 质量: chosen (原始) 0.9, rejected (其他角色) 0.3-0.6
        """
        preference_pairs = []
        
        # Step 1: 按 prompt → 角色 聚类
        prompt_chars = defaultdict(lambda: defaultdict(list))
        for item in data:
            prompt = item['prompt']
            char = item['character']
            prompt_chars[prompt][char].append(item)
        
        print(f"\n检测到 {len(prompt_chars)} 个独特 prompts")
        print(f"角色数量: {len(set(item['character'] for item in data))}")
        
        # Step 2: 为每个 prompt 的每个角色生成偏好对
        for prompt, char_responses in tqdm(prompt_chars.items(), desc="生成角色错配对"):
            if len(char_responses) < 2:
                # 需要至少 2 个角色才能形成对比
                continue
            
            # 按角色降序排列 (多回复的角色多生成偏好对)
            characters = list(char_responses.keys())
            random.shuffle(characters)
            
            # 对每个角色，选择其他角色作为 rejected
            for chosen_char in characters:
                chosen_responses = char_responses[chosen_char]
                
                # 选择该角色的最好回复作为 chosen
                chosen = random.choice(chosen_responses)
                
                # 从其他角色选择 rejected
                other_chars = [c for c in characters if c != chosen_char]
                if not other_chars:
                    continue
                
                for rejected_char in random.sample(other_chars, min(2, len(other_chars))):
                    rejected = random.choice(char_responses[rejected_char])
                    
                    pair = {
                        'prompt': prompt,
                        'chosen': chosen['response'],
                        'rejected': rejected['response'],
                        'chosen_character': chosen_char,
                        'rejected_character': rejected_char,
                        'language': chosen.get('language', 'unknown'),
                        'quality_chosen': 0.85,    # 原始角色的回复
                        'quality_rejected': 0.35,  # 其他角色的回复 (角色不匹)
                        'reason': f"角色错配: {chosen_char} vs {rejected_char}",
                    }
                    preference_pairs.append(pair)
        
        return preference_pairs
    
    def generate_by_multi_language(self, data: List[Dict]) -> List[Dict]:
        """
        【方案 B】语言错配: 同一语义，不同语言竞争
        
        核心思路:
        1. 按含义相近的 prompt 聚类
        2. 不同语言版本 = chosen + rejected
        3. 质量: 语言匹配 0.9 vs 不匹配 0.2
        """
        preference_pairs = []
        
        # 按语言分组
        lang_data = defaultdict(lambda: defaultdict(list))
        for item in data:
            lang = item.get('language', 'unknown')
            char = item['character']
            lang_data[lang][char].append(item)
        
        languages = list(lang_data.keys())
        print(f"\n检测到 {len(languages)} 种语言")
        
        # 为每种语言生成偏好对
        for lang, char_data in tqdm(lang_data.items(), desc="生成语言错配对"):
            if not char_data:
                continue
            
            # 获取该语言的所有回复
            items = []
            for char_items in char_data.values():
                items.extend(char_items)
            
            if len(items) < 2:
                continue
            
            # 随机配对
            sampled = random.sample(items, min(100, len(items)))
            
            for i in range(0, len(sampled) - 1, 2):
                chosen = sampled[i]
                rejected = sampled[i + 1]
                
                # 选择不同的语言作为 rejected
                other_langs = [l for l in languages if l != lang]
                if not other_langs:
                    continue
                
                rejected_lang = random.choice(other_langs)
                rejected_items = lang_data[rejected_lang]
                rejected_sample = []
                for char_items in rejected_items.values():
                    rejected_sample.extend(char_items)
                
                if rejected_sample:
                    rejected = random.choice(rejected_sample)
                    
                    pair = {
                        'prompt': chosen['prompt'],
                        'chosen': chosen['response'],
                        'rejected': rejected['response'],
                        'chosen_language': lang,
                        'rejected_language': rejected_lang,
                        'character': chosen['character'],
                        'quality_chosen': 0.90,     # 语言匹配
                        'quality_rejected': 0.20,  # 语言不匹
                        'reason': f"语言错配: {lang} vs {rejected_lang}",
                    }
                    preference_pairs.append(pair)
        
        return preference_pairs
    
    def generate_by_length_hierarchy(self, data: List[Dict]) -> List[Dict]:
        """
        【方案 C】长度/信息层级: 对每个响应生成截断版本
        
        核心思路:
        1. 完整回复 = chosen (0.85)
        2. 截断版本 = rejected (0.4-0.6)
        """
        preference_pairs = []
        
        for item in tqdm(data, desc="生成长度层级对"):
            response = item['response']
            prompt = item['prompt']
            
            if len(response) < 50:  # 太短的跳过
                continue
            
            # 生成不同长度的截断版本
            full_len = len(response)
            cutoff_points = [
                (0.8, 0.75),   # 保留 80%, 质量 0.75
                (0.6, 0.55),   # 保留 60%, 质量 0.55
                (0.4, 0.35),   # 保留 40%, 质量 0.35
            ]
            
            for ratio, quality in cutoff_points:
                cutoff_pos = int(full_len * ratio)
                truncated = response[:cutoff_pos].rstrip() + "..."
                
                pair = {
                    'prompt': prompt,
                    'chosen': response,
                    'rejected': truncated,
                    'character': item['character'],
                    'language': item.get('language', 'unknown'),
                    'quality_chosen': 0.85,
                    'quality_rejected': quality,
                    'reason': f"信息完整性: 100% vs {int(ratio*100)}%",
                }
                preference_pairs.append(pair)
        
        return preference_pairs
    
    def generate_by_combined(self, data: List[Dict]) -> List[Dict]:
        """
        【方案 D】混合: 角色+语言同时错配
        """
        # 此为最强方案，综合 A 和 B
        pairs_role = self.generate_by_multi_role(data)
        pairs_lang = self.generate_by_multi_language(data)
        
        # 混合 50% 角色错配 + 50% 语言错配
        combined = pairs_role[::2] + pairs_lang[::2]
        return combined[:min(len(pairs_role), len(pairs_lang)) * 2]
    
    def save_data(self, data: List[Dict], output_file: str) -> None:
        """保存为 JSONL"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def process(self, input_file: str, output_file: str, method: str = 'multi_role') -> None:
        """主流程"""
        print(f"[1/4] 加载数据: {input_file}")
        data = self.load_data(input_file)
        print(f"  ✓ 加载 {len(data)} 条数据")
        
        print(f"\n[2/4] 生成偏好对 (方法: {method})")
        if method == 'multi_role':
            pairs = self.generate_by_multi_role(data)
        elif method == 'multi_language':
            pairs = self.generate_by_multi_language(data)
        elif method == 'multi_length':
            pairs = self.generate_by_length_hierarchy(data)
        elif method == 'multi_combined':
            pairs = self.generate_by_combined(data)
        else:
            raise ValueError(f"未知方法: {method}")
        
        print(f"  ✓ 生成 {len(pairs)} 个偏好对")
        
        # 统计
        print(f"\n[3/4] 统计信息:")
        
        if 'chosen_character' in pairs[0]:
            char_pairs = defaultdict(int)
            for pair in pairs:
                char_pairs[pair['chosen_character']] += 1
            
            print(f"  【按角色统计 (chosen)】")
            for char, count in sorted(char_pairs.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"    - {char}: {count} 对")
            
            if len(char_pairs) > 10:
                print(f"    ... 及其他 {len(char_pairs)-10} 个角色")
        
        if 'reason' in pairs[0]:
            reasons = defaultdict(int)
            for pair in pairs:
                reasons[pair['reason'][:30]] += 1
            print(f"  【错配类型分布】")
            for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"    - {reason}: {count} 对")
        
        print(f"\n[4/4] 保存数据: {output_file}")
        self.save_data(pairs, output_file)
        print(f"  ✓ 完成!")
        
        # 示例
        print(f"\n【示例对】:")
        for i, pair in enumerate(pairs[:2]):
            print(f"\n样本 {i+1}:")
            print(f"  Prompt: {pair['prompt'][:50]}...")
            print(f"  Chosen: {pair['chosen'][:50]}...")
            print(f"  Rejected: {pair['rejected'][:50]}...")
            if 'reason' in pair:
                print(f"  原因: {pair['reason']}")
            print(f"  质量: {pair['quality_chosen']:.2f} vs {pair['quality_rejected']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='多角色多语言 Preference Pair 生成'
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
        '--method',
        type=str,
        default='multi_role',
        choices=['multi_role', 'multi_language', 'multi_length', 'multi_combined'],
        help='生成方法'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    
    args = parser.parse_args()
    
    generator = AdvancedPreferenceDataGenerator(seed=args.seed)
    generator.process(
        input_file=args.input,
        output_file=args.output,
        method=args.method
    )


if __name__ == '__main__':
    main()
