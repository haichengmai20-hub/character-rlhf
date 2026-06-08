#!/usr/bin/env python3
"""
阶段 3: Actor 模型构造与 RL 强化学习 (GRPO)
基于 SFT 模型，使用 RM 指导进行强化学习训练

Architecture:
  策略模型 (Actor): SFT Model Backbone + RL-LoRA (可训练)
  参考模型 (Reference): SFT Model (完全冻结，仅推理)
  奖励模型 (Reward Model): RM 产物 (冻结，仅推理)

GRPO 训练流程:
  1. Actor 根据 prompt 生成回复 (Group 内生成多个回复)
  2. Reward Model 评估每个回复
  3. 计算相对奖励 (组内 rank)
  4. 计算 RL Loss (GRPO) + KL 散度约束
  5. 更新 Actor LoRA 参数

Usage:
    python train_actor_model.py \
        --sft_model_path /path/to/sft_model \
        --rm_model_path /path/to/rm_model \
        --prompt_data /path/to/prompts.jsonl \
        --output_dir /path/to/actor_output
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import logging
import argparse
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(rankId}s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ActorDataset(Dataset):
    """Actor 训练数据集 (仅需要 prompts)"""
    
    def __init__(self, 
                 data_file: str,
                 tokenizer,
                 max_length: int = 1536):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompts = []
        
        # 加载 prompts
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                prompt = item.get('prompt', '')
                if prompt:
                    self.prompts.append(prompt)
        
        logger.info(f"Loaded {len(self.prompts)} prompts for Actor training")
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        
        tokens = self.tokenizer(
            prompt,
            max_length=self.max_length // 2,  # 为生成留出空间
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'prompt': prompt
        }


class ActorTrainer:
    """Actor 强化学习训练器 (GRPO)"""
    
    def __init__(self, args):
        self.args = args
        
        # 加载模型
        logger.info("Loading models...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, trust_remote_code=True)
        
        # 策略模型 (Actor)
        self.actor = AutoModelForCausalLM.from_pretrained(
            args.sft_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            load_in_4bit=True  # 减少显存占用
        )
        
        # 参考模型 (冻结)
        self.reference = AutoModelForCausalLM.from_pretrained(
            args.sft_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            load_in_4bit=True
        )
        
        # RM 模型 (冻结)
        if os.path.exists(args.rm_model_path):
            self.rm = AutoModelForCausalLM.from_pretrained(
                args.rm_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                load_in_4bit=True
            )
            score_head_path = os.path.join(args.rm_model_path, "score_head.pt")
            self.score_head = self._load_score_head(score_head_path)
        else:
            logger.warning("RM model not found, using reward data directly")
        
        # 冻结参考模型和RM
        for param in self.reference.parameters():
            param.requires_grad = False
        if hasattr(self, 'rm'):
            for param in self.rm.parameters():
                param.requires_grad = False
        
        # 添加 RL-LoRA 到 Actor
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.actor = get_peft_model(self.actor, lora_config)
        
        logger.info("✓ Models loaded")
    
    def _load_score_head(self, path):
        """加载 RM 得分头"""
        hidden_size = self.rm.config.hidden_size
        from train_rm_model import RMScoreHead
        score_head = RMScoreHead(hidden_size)
        score_head.load_state_dict(torch.load(path, map_location="cpu"))
        return score_head.to(self.rm.device)
    
    def generate_completions(self, prompts: List[str], num_samples: int = 4) -> List[Dict]:
        """
        生成多个回复 (用于 GRPO 分组比较)
        
        Args:
            prompts: 输入 prompts
            num_samples: 每个 prompt 生成的回复数量
        
        Returns:
            completion_groups: 分组的完成数据
        """
        completions_groups = []
        
        for prompt in prompts:
            group = {'prompt': prompt, 'completions': []}
            
            for _ in range(num_samples):
                # 使用 Actor 生成
                inputs = self.tokenizer(
                    prompt,
                    return_tensors='pt'
                ).to(self.actor.device)
                
                with torch.no_grad():
                    outputs = self.actor.generate(
                        **inputs,
                        max_new_tokens=self.args.max_new_tokens,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        num_return_sequences=1
                    )
                
                completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # 移除 prompt
                completion = completion[len(prompt):].strip()
                group['completions'].append(completion)
            
            completions_groups.append(group)
        
        return completions_groups
    
    def compute_rewards(self, completion_groups: List[Dict]) -> List[Dict]:
        """使用 RM 计算奖励"""
        
        for group in completion_groups:
            rewards = []
            
            for completion in group['completions']:
                # 使用 RM 评估
                text = group['prompt'] + completion
                inputs = self.tokenizer(
                    text,
                    max_length=self.args.max_length,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.rm.device)
                
                with torch.no_grad():
                    outputs = self.rm(**inputs, output_hidden_states=True)
                    reward = self.score_head(outputs.hidden_states[-1]).item()
                
                rewards.append(reward)
            
            group['rewards'] = rewards
        
        return completion_groups
    
    def compute_grpo_loss(self, completion_groups: List[Dict]) -> torch.Tensor:
        """
        计算 GRPO Loss
        
        GRPO = max_a E_tau[sign(r_best - r_a) * log π_actor(a|τ)]
        + β * KL(π_actor || π_ref)
        """
        
        total_loss = 0
        num_samples = 0
        
        for group in completion_groups:
            prompt = group['prompt']
            completions = group['completions']
            rewards = torch.tensor(group['rewards'])
            
            # 排序并获取最佳/最差
            sorted_idx = torch.argsort(rewards, descending=True)
            best_idx = sorted_idx[0].item()
            
            # 计算相对奖励
            for i, completion in enumerate(completions):
                if i == best_idx:
                    continue  # 最佳不参与训练
                
                # Actor logits
                text = prompt + completion
                actor_inputs = self.tokenizer(
                    text,
                    max_length=self.args.max_length,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.actor.device)
                
                actor_outputs = self.actor(**actor_inputs)
                actor_logits = actor_outputs.logits
                
                # Reference logits (KL 约束)
                ref_inputs = self.tokenizer(
                    text,
                    max_length=self.args.max_length,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.reference.device)
                
                with torch.no_grad():
                    ref_outputs = self.reference(**ref_inputs)
                    ref_logits = ref_outputs.logits
                
                # LogProb
                actor_log_probs = F.log_softmax(actor_logits, dim=-1)
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                
                # Policy Gradient Loss
                reward_diff = rewards[best_idx] - rewards[i]
                policy_loss = -reward_diff * actor_log_probs.sum()
                
                # KL 散度 (约束 policy 不偏离 reference)
                kl_loss = torch.nn.functional.kl_div(
                    actor_log_probs,
                    torch.exp(ref_log_probs),
                    reduction='batchmean'
                )
                
                # 总 Loss
                loss = policy_loss + self.args.beta_kl * kl_loss
                total_loss += loss
                num_samples += 1
        
        return total_loss / max(num_samples, 1)
    
    def train(self):
        """主训练循环"""
        
        logger.info("="*70)
        logger.info("Starting Actor RL Training (GRPO)")
        logger.info("="*70)
        
        # 加载数据
        dataset = ActorDataset(
            data_file=self.args.prompt_data,
            tokenizer=self.tokenizer,
            max_length=self.args.max_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # 优化器
        optimizer = AdamW(self.actor.parameters(), lr=self.args.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(len(dataloader) * self.args.warmup_ratio),
            num_training_steps=len(dataloader) * self.args.num_train_epochs
        )
        
        # 训练循环
        self.actor.train()
        global_step = 0
        
        for epoch in range(self.args.num_train_epochs):
            epoch_loss = 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.num_train_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                prompts = batch['prompt']
                
                # 1. 生成多个回复 (Group)
                completion_groups = self.generate_completions(
                    prompts,
                    num_samples=self.args.num_samples_per_prompt
                )
                
                # 2. 计算奖励
                completion_groups = self.compute_rewards(completion_groups)
                
                # 3. 计算 GRPO Loss
                loss = self.compute_grpo_loss(completion_groups)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                avg_loss = epoch_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                
                # 保存 checkpoint
                if global_step % self.args.save_steps == 0:
                    save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                    logger.info(f"Saving Actor checkpoint to {save_path}")
                    self.actor.save_pretrained(save_path)
                    self.tokenizer.save_pretrained(save_path)
            
            logger.info(f"Epoch {epoch+1} - Average RL Loss: {epoch_loss/len(dataloader):.4f}")
        
        # 保存最终模型
        logger.info(f"Saving final Actor model to {self.args.output_dir}")
        self.actor.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)
        
        logger.info("✓ Actor RL Training Complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Actor with RL (GRPO)")
    
    # Model paths
    parser.add_argument("--sft_model_path", type=str, required=True,
                       help="Path to SFT model (Actor 初始化)")
    parser.add_argument("--rm_model_path", type=str, required=True,
                       help="Path to Reward Model (RM 初始化)")
    
    # Data
    parser.add_argument("--prompt_data", type=str, required=True,
                       help="Prompt data file (仅需要 prompt 字段)")
    
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=64,
                       help="RL-LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # RL 参数
    parser.add_argument("--num_samples_per_prompt", type=int, default=4,
                       help="GRPO 生成的回复数量 (group size)")
    parser.add_argument("--beta_kl", type=float, default=0.1,
                       help="KL 散度系数")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="生成最大长度")
    
    # Training
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Data
    parser.add_argument("--max_length", type=int, default=1536)
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for Actor model")
    parser.add_argument("--save_steps", type=int, default=100)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练
    trainer = ActorTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
