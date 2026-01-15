"""
Pretraining Dataset
Convert text data into a format usable by the model
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional


class PretrainDataset(Dataset):
    """
    Pretraining Dataset
    Split long text into fixed-length sequences
    """
    
    def __init__(
        self,
        data_path: Path,
        tokenizer,
        max_seq_len: int = 512,
        stride: int = 256,  # Sliding window stride, can be overlapping
    ):
        """
        Args:
            data_path: Path to the text file
            tokenizer: LLMTokenizer instance
            max_seq_len: Maximum sequence length
            stride: Sliding window stride
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride
        
        # Read and encode all text
        print(f"Loading data: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Encode the entire text (do not add special tokens)
        print("Encoding text...")
        self.all_tokens = tokenizer.encode(text, add_special_tokens=False)
        print(f"Total number of tokens: {len(self.all_tokens):,}")
        
        # Calculate number of samples
        self.num_samples = max(1, (len(self.all_tokens) - max_seq_len) // stride + 1)
        print(f"Number of samples: {self.num_samples:,}")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        start = idx * self.stride
        end = start + self.max_seq_len
        
        # Get token sequence
        tokens = self.all_tokens[start:end]
        
        # If less than max_seq_len, pad with pad token
        if len(tokens) < self.max_seq_len:
            padding = [self.tokenizer.pad_token_id] * (self.max_seq_len - len(tokens))
            attention_mask = [1] * len(tokens) + [0] * len(padding)
            tokens = tokens + padding
        else:
            attention_mask = [1] * self.max_seq_len
        
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(tokens, dtype=torch.long),  # Autoregressive: labels = input_ids
        }


def create_dataloaders(
    train_path: Path,
    val_path: Optional[Path],
    tokenizer,
    max_seq_len: int = 512,
    batch_size: int = 8,
    num_workers: int = 0,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders
    """
    train_dataset = PretrainDataset(train_path, tokenizer, max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = None
    if val_path and val_path.exists():
        val_dataset = PretrainDataset(val_path, tokenizer, max_seq_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return train_loader, val_loader