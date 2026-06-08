#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 Actor 训练数据 (Prompts Only)

用法:
  python generate_actor_prompts.py \
    --input your_data.jsonl \
    --output actor_prompts.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm


class ActorPromptGenerator:
    """生成 Actor 强化学习的提示数据"""
    
    def __init__(self):
        pass
    
    def load_data(self, input_file: str) -> List[Dict]:
        """加载输入数据"""
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def extract_prompts(self, data: List[Dict]) -> List[Dict]:
        """
        从原始数据提取 prompts
        
        支持多种数据格式:
        1. {"prompt": "...", "response": "...", ...}
        2. {"instruction": "...", "input": "...", "output": "...", ...}
        3. {"question": "...", "answer": "...", ...}
        4. {"text": "...", ...}
        """
        prompts = []
        seen_prompts = set()
        
        for item in tqdm(data, desc="提取 Prompts"):
            # 尝试多个字段名来获取 prompt
            prompt = (
                item.get('prompt') or
                item.get('instruction') or
                item.get('question') or
                item.get('query') or
                ""
            )
            
            if not prompt:
                continue
            
            # 去重
            if prompt in seen_prompts:
                continue
            
            seen_prompts.add(prompt)
            
            prompt_item = {
                'prompt': prompt,
                'character': item.get('character', 'default'),
                'language': item.get('language', 'unknown'),
                'category': item.get('category', 'general'),
                'difficulty': item.get('difficulty', 'medium'),
            }
            
            prompts.append(prompt_item)
        
        return prompts
    
    def filter_and_balance(
        self,
        prompts: List[Dict],
        min_prompts_per_char: int = 20,
    ) -> List[Dict]:
        """
        过滤和平衡数据
        
        确保:
        1. 每个字符有足够的 prompts
        2. 字符分布相对均匀
        """
        # 按字符分组
        char_prompts = defaultdict(list)
        for prompt in prompts:
            char = prompt['character']
            char_prompts[char].append(prompt)
        
        print(f"\n字符分布 (过滤前):")
        for char in sorted(char_prompts.keys()):
            print(f"  - {char}: {len(char_prompts[char])} 个 prompts")
        
        # 过滤: 每个字符至少要有 min_prompts_per_char 个
        filtered = []
        for char, char_data in char_prompts.items():
            if len(char_data) >= min_prompts_per_char:
                filtered.extend(char_data)
                print(f"  ✓ {char}: 保留 {len(char_data)} 个")
            else:
                print(f"  ✗ {char}: 仅 {len(char_data)} 个，低于阈值 {min_prompts_per_char}")
        
        print(f"\n过滤后: {len(filtered)} 个 prompts")
        return filtered
    
    def add_conversation_context(self, prompts: List[Dict]) -> List[Dict]:
        """
        添加对话上下文
        
        某些 prompts 可能需要系统消息作为上下文
        """
        enhanced = []
        
        for prompt_item in prompts:
            char = prompt_item['character']
            
            # 这里可以集成 TOP20 字符系统提示
            # 示例: char_system_prompts[char]
            
            enhanced_item = {
                'prompt': prompt_item['prompt'],
                'character': char,
                'language': prompt_item['language'],
                'category': prompt_item['category'],
                'difficulty': prompt_item['difficulty'],
                # 可选: system_message, context 等
            }
            enhanced.append(enhanced_item)
        
        return enhanced
    
    def save_data(self, data: List[Dict], output_file: str) -> None:
        """保存为 JSONL"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def process(
        self,
        input_file: str,
        output_file: str,
        min_prompts_per_char: int = 20,
    ) -> None:
        """主流程"""
        print(f"[1/4] 加载数据: {input_file}")
        data = self.load_data(input_file)
        print(f"  ✓ 加载 {len(data)} 条数据")
        
        print(f"\n[2/4] 提取 Prompts")
        prompts = self.extract_prompts(data)
        print(f"  ✓ 提取 {len(prompts)} 个独特 prompts")
        
        print(f"\n[3/4] 过滤和平衡")
        prompts = self.filter_and_balance(prompts, min_prompts_per_char)
        
        # 添加上下文
        prompts = self.add_conversation_context(prompts)
        
        print(f"\n[4/4] 保存数据: {output_file}")
        self.save_data(prompts, output_file)
        print(f"  ✓ 完成!")
        
        # 统计
        print(f"\n数据统计:")
        char_counts = defaultdict(int)
        lang_counts = defaultdict(int)
        for p in prompts:
            char_counts[p['character']] += 1
            lang_counts[p['language']] += 1
        
        print(f"\n按字符统计:")
        for char in sorted(char_counts.keys()):
            print(f"  - {char}: {char_counts[char]}")
        
        print(f"\n按语言统计:")
        for lang in sorted(lang_counts.keys()):
            print(f"  - {lang}: {lang_counts[lang]}")
        
        # 示例
        print(f"\n示例 Prompts:")
        for i, p in enumerate(prompts[:3]):
            print(f"\n样本 {i+1}:")
            print(f"  Prompt: {p['prompt'][:100]}...")
            print(f"  Character: {p['character']}")
            print(f"  Language: {p['language']}")


def main():
    parser = argparse.ArgumentParser(
        description='生成 Actor 强化学习的 Prompt 数据'
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
        help='输出 prompts 文件'
    )
    parser.add_argument(
        '--min-prompts-per-char',
        type=int,
        default=20,
        help='每个字符最少的 prompts 数'
    )
    
    args = parser.parse_args()
    
    generator = ActorPromptGenerator()
    generator.process(
        input_file=args.input,
        output_file=args.output,
        min_prompts_per_char=args.min_prompts_per_char,
    )


if __name__ == '__main__':
    main()
