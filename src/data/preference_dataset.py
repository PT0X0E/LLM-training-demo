"""
Preference dataset (for DPO training)
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional


class PreferenceDataset(Dataset):
    """
    DPO preference dataset
    
    Data format (JSONL):
    {"prompt": "question", "chosen": "better answer", "rejected": "worse answer"}
    """
    
    def __init__(
        self,
        data_path: Path,
        tokenizer,
        max_seq_len: int = 512,
    ):
        """
        Args:
            data_path: JSONL data file path
            tokenizer: LLMTokenizer instance
            max_seq_len: maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Load data
        self.data = []
        print(f"Loading preference data: {data_path}")
        
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
        
        print(f"Number of samples loaded: {len(self.data)}")
    
    def _tokenize(self, prompt: str, response: str) -> dict:
        """
        Encode prompt + response
        Return input_ids, attention_mask, labels
        """
        # Build full text
        full_text = f"User: {prompt}\nAssistant: {response}"
        prompt_text = f"User: {prompt}\nAssistant: "
        
        # Encode
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        
        # Add EOS
        full_ids = full_ids + [self.tokenizer.eos_token_id]
        
        # Create labels (only compute loss for response part)
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        
        # Truncate
        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        
        # Padding
        attention_mask = [1] * len(full_ids)
        padding_len = self.max_seq_len - len(full_ids)
        
        if padding_len > 0:
            full_ids = full_ids + [self.tokenizer.pad_token_id] * padding_len
            labels = labels + [-100] * padding_len
            attention_mask = attention_mask + [0] * padding_len
        
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # Encode chosen and rejected separately
        chosen_tokens = self._tokenize(prompt, chosen)
        rejected_tokens = self._tokenize(prompt, rejected)
        
        return {
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "chosen_labels": chosen_tokens["labels"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
            "rejected_labels": rejected_tokens["labels"],
        }


def create_preference_dataloaders(
    train_path: Path,
    val_path: Optional[Path],
    tokenizer,
    max_seq_len: int = 512,
    batch_size: int = 4,
    num_workers: int = 0,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """Create preference data loaders"""
    
    train_dataset = PreferenceDataset(train_path, tokenizer, max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = None
    if val_path and val_path.exists():
        val_dataset = PreferenceDataset(val_path, tokenizer, max_seq_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return train_loader, val_loader