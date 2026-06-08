#!/usr/bin/env python3
"""
阶段 2: Reward Model (RM) 训练
基于 SFT 模型初始化，训练奖励模型用于评估回复质量

Architecture:
  SFT Model (冻结 Backbone) + RM-LoRA (可训练)
  └─ 输出: score head (预测回复质量分数)

Usage:
    python train_rm_model.py \
        --sft_model_path /path/to/sft_model \
        --train_file /path/to/train_data.jsonl \
        --output_dir /path/to/rm_output \
        --lora_rank 64
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import logging
import argparse


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RMDataset(Dataset):
    """奖励模型训练数据集 (Preference Pairs)"""
    
    def __init__(self, 
                 data_file: str,
                 tokenizer,
                 max_length: int = 1536):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # 加载成对比较数据
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                
                # 需要格式: prompt, chosen (高质量回复), rejected (低质量回复)
                if 'prompt' in item and 'chosen' in item and 'rejected' in item:
                    self.samples.append(item)
        
        logger.info(f"Loaded {len(self.samples)} preference pairs for RM training")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        prompt = item.get('prompt', '')
        chosen = item.get('chosen', '')
        rejected = item.get('rejected', '')
        
        # 编码chosen和rejected回复
        chosen_input = f"{prompt}{chosen}"
        rejected_input = f"{prompt}{rejected}"
        
        chosen_tokens = self.tokenizer(
            chosen_input,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        rejected_tokens = self.tokenizer(
            rejected_input,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_tokens['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_tokens['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_tokens['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_tokens['attention_mask'].squeeze(0),
        }


class RMScoreHead(torch.nn.Module):
    """奖励模型得分头 (线性层 + sigmoid)"""
    
    def __init__(self, hidden_size: int = 2048):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, last_hidden_states):
        # last_hidden_states: [batch_size, seq_len, hidden_size]
        # 使用最后一个token的隐藏状态
        scores = self.linear(last_hidden_states[:, -1, :])  # [batch_size, 1]
        return scores.squeeze(-1)  # [batch_size]


def train_rm_model(args):
    """训练奖励模型"""
    
    logger.info("="*70)
    logger.info("Starting Reward Model (RM) Training")
    logger.info("="*70)
    
    # 1. 加载SFT模型
    logger.info(f"Loading SFT model from {args.sft_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 2. 冻结backbone，仅训练LoRA
    for param in base_model.parameters():
        param.requires_grad = False
    
    # 3. 添加LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(base_model, lora_config)
    logger.info(f"Added LoRA (rank={args.lora_rank}) to model")
    
    # 4. 添加得分头 (可训练)
    hidden_size = base_model.config.hidden_size
    score_head = RMScoreHead(hidden_size).to(base_model.device)
    
    # 5. 加载数据
    logger.info(f"Loading RM training data from {args.train_file}")
    dataset = RMDataset(
        data_file=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # 6. 优化器和调度器
    trainable_params = list(model.parameters()) + list(score_head.parameters())
    optimizer = AdamW(trainable_params, lr=args.learning_rate)
    
    total_steps = len(dataloader) * args.num_train_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )
    
    # 7. 训练循环
    model.train()
    score_head.train()
    
    global_step = 0
    for epoch in range(args.num_train_epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # 获取chosen和rejected的得分
            chosen_ids = batch['chosen_input_ids'].to(base_model.device)
            chosen_mask = batch['chosen_attention_mask'].to(base_model.device)
            rejected_ids = batch['rejected_input_ids'].to(base_model.device)
            rejected_mask = batch['rejected_attention_mask'].to(base_model.device)
            
            # Forward pass
            with torch.no_grad():
                chosen_output = base_model(
                    input_ids=chosen_ids,
                    attention_mask=chosen_mask,
                    output_hidden_states=True
                )
                rejected_output = base_model(
                    input_ids=rejected_ids,
                    attention_mask=rejected_mask,
                    output_hidden_states=True
                )
            
            # 计算得分
            chosen_score = score_head(chosen_output.hidden_states[-1])
            rejected_score = score_head(rejected_output.hidden_states[-1])
            
            # Ranking Loss: chosen > rejected
            # 使用 margin ranking loss
            loss = F.margin_ranking_loss(
                chosen_score, 
                rejected_score,
                torch.ones_like(chosen_score),
                margin=args.margin
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            avg_loss = epoch_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            # 保存checkpoint
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                logger.info(f"Saving checkpoint to {save_path}")
                model.save_pretrained(save_path)
                score_head_path = os.path.join(save_path, "score_head.pt")
                torch.save(score_head.state_dict(), score_head_path)
                tokenizer.save_pretrained(save_path)
        
        logger.info(f"Epoch {epoch+1} - Average Loss: {epoch_loss/len(dataloader):.4f}")
    
    # 8. 保存最终模型
    logger.info(f"Saving final RM model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    score_head_path = os.path.join(args.output_dir, "score_head.pt")
    torch.save(score_head.state_dict(), score_head_path)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("✓ RM Training Complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Reward Model (RM)")
    
    # Model
    parser.add_argument("--sft_model_path", type=str, required=True,
                       help="Path to SFT model (必须)")
    
    # Data
    parser.add_argument("--train_file", type=str, required=True,
                       help="Training data file (format: prompt, chosen, rejected)")
    
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=64,
                       help="LoRA rank (建议 >= 64)")
    parser.add_argument("--lora_alpha", type=int, default=128,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")
    
    # Training
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=1.0,
                       help="Margin in ranking loss")
    
    # Data
    parser.add_argument("--max_length", type=int, default=1536)
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for RM model")
    parser.add_argument("--save_steps", type=int, default=200)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_rm_model(args)


if __name__ == "__main__":
    main()
