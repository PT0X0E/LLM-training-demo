"""
SFT instruction fine-tuning dataset
Supports multiple dialogue formats
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional


class SFTDataset(Dataset):
    """
    SFT instruction fine-tuning dataset
    
    Data format (JSONL):
    {"instruction": "question", "input": "optional input", "output": "answer"}
    or
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    
    def __init__(
        self,
        data_path: Path,
        tokenizer,
        max_seq_len: int = 512,
        prompt_template: str = "alpaca",  # alpaca, chatml
    ):
        """
        Args:
            data_path: JSONL data file path
            tokenizer: LLMTokenizer instance
            max_seq_len: maximum sequence length
            prompt_template: prompt template format
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template
        
        # Load data
        self.data = []
        print(f"Loading SFT data: {data_path}")
        
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
        
        print(f"Number of samples loaded: {len(self.data)}")
    
    def format_alpaca(self, item: dict) -> tuple[str, str]:
        """
        Alpaca format template
        """
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        if input_text:
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        else:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
        
        return prompt, output
    
    def format_chatml(self, item: dict) -> tuple[str, str]:
        """
        ChatML format template (OpenAI style)
        """
        if "messages" in item:
            messages = item["messages"]
        else:
            # Convert alpaca format to messages
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")
            
            user_content = instruction
            if input_text:
                user_content += f"\n\n{input_text}"
            
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ]
        
        # Build prompt (excluding the last assistant reply)
        prompt_parts = []
        response = ""
        
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                if i == len(messages) - 1:
                    # Last assistant message as response
                    prompt_parts.append("<|im_start|>assistant\n")
                    response = content
                else:
                    prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        prompt = "\n".join(prompt_parts)
        return prompt, response
    
    def format_simple(self, item: dict) -> tuple[str, str]:
        """
        Simple format template
        """
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        if input_text:
            prompt = f"User: {instruction}\n{input_text}\nAssistant: "
        else:
            prompt = f"User: {instruction}\nAssistant: "
        
        return prompt, output
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        
        # Format according to template
        if self.prompt_template == "alpaca":
            prompt, response = self.format_alpaca(item)
        elif self.prompt_template == "chatml":
            prompt, response = self.format_chatml(item)
        else:
            prompt, response = self.format_simple(item)
        
        # Encode
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        eos_id = self.tokenizer.eos_token_id
        
        # Concatenate: prompt + response + eos
        input_ids = prompt_ids + response_ids + [eos_id]
        
        # Create labels (only compute loss for response part)
        # Mark prompt part with -100, do not compute loss
        labels = [-100] * len(prompt_ids) + response_ids + [eos_id]
        
        # Truncate
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        
        # Padding
        attention_mask = [1] * len(input_ids)
        padding_len = self.max_seq_len - len(input_ids)
        
        if padding_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
            labels = labels + [-100] * padding_len  # Do not compute loss for padding
            attention_mask = attention_mask + [0] * padding_len
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def create_sft_dataloaders(
    train_path: Path,
    val_path: Optional[Path],
    tokenizer,
    max_seq_len: int = 512,
    batch_size: int = 4,
    prompt_template: str = "alpaca",
    num_workers: int = 0,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """Create SFT data loaders"""
    
    train_dataset = SFTDataset(train_path, tokenizer, max_seq_len, prompt_template)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = None
    if val_path and val_path.exists():
        val_dataset = SFTDataset(val_path, tokenizer, max_seq_len, prompt_template)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return train_loader, val_loader