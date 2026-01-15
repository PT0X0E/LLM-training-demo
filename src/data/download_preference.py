"""
Download preference datasets (for RLHF/DPO post-training)
"""

import json
from pathlib import Path
from datasets import load_dataset


DATA_DIR = Path(__file__).parent.parent.parent / "data"


def download_hh_rlhf(save_dir: Path = None, num_samples: int = 10000):
    """
    Download Anthropic HH-RLHF dataset
    - Human preference dialogue data
    - Contains helpful and harmless subsets
    - Classic RLHF dataset
    """
    save_dir = save_dir or DATA_DIR / "preference"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading Anthropic HH-RLHF dataset...")
    
    # Download helpful subset
    dataset = load_dataset(
        "Anthropic/hh-rlhf",
        data_dir="helpful-base",
        split="train"
    )
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Convert format
    train_data = []
    for item in dataset:
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # Extract prompt (shared prefix of both)
        # HH-RLHF format: "\n\nHuman: xxx\n\nAssistant: xxx"
        # Find the last Human: as prompt
        if "\n\nHuman:" in chosen and "\n\nAssistant:" in chosen:
            # Find the last conversation turn
            parts = chosen.rsplit("\n\nAssistant:", 1)
            if len(parts) == 2:
                prompt = parts[0] + "\n\nAssistant:"
                chosen_response = parts[1].strip()
                
                # Extract corresponding response from rejected
                rejected_parts = rejected.rsplit("\n\nAssistant:", 1)
                if len(rejected_parts) == 2:
                    rejected_response = rejected_parts[1].strip()
                    
                    train_data.append({
                        "prompt": prompt,
                        "chosen": chosen_response,
                        "rejected": rejected_response,
                    })
    
    # Split into train and validation sets
    split_idx = int(len(train_data) * 0.95)
    train_split = train_data[:split_idx]
    val_split = train_data[split_idx:]
    
    # Save
    with open(save_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for item in train_split:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(save_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for item in val_split:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"HH-RLHF dataset download complete!")
    print(f"  Train set: {save_dir / 'train.jsonl'} ({len(train_split)} samples)")
    print(f"  Val set: {save_dir / 'val.jsonl'} ({len(val_split)} samples)")


def download_ultrafeedback(save_dir: Path = None, num_samples: int = 10000):
    """
    Download UltraFeedback dataset
    - Response comparisons from multiple models
    - Contains ratings and preferences
    - Suitable for DPO training
    """
    save_dir = save_dir or DATA_DIR / "preference"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading UltraFeedback dataset...")
    
    dataset = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs"
    )
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Convert format
    train_data = []
    for item in dataset:
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # Extract response content
        chosen_content = chosen[1]["content"] if len(chosen) > 1 else ""
        rejected_content = rejected[1]["content"] if len(rejected) > 1 else ""
        
        if chosen_content and rejected_content:
            train_data.append({
                "prompt": prompt,
                "chosen": chosen_content,
                "rejected": rejected_content,
            })
    
    # Split
    split_idx = int(len(train_data) * 0.95)
    train_split = train_data[:split_idx]
    val_split = train_data[split_idx:]
    
    # Save
    with open(save_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for item in train_split:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(save_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for item in val_split:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"UltraFeedback dataset download complete!")
    print(f"  Train set: {save_dir / 'train.jsonl'} ({len(train_split)} samples)")
    print(f"  Val set: {save_dir / 'val.jsonl'} ({len(val_split)} samples)")


def download_shp(save_dir: Path = None, num_samples: int = 10000):
    """
    Download Stanford Human Preferences (SHP) dataset
    - Real human preferences from Reddit
    - Preferences based on upvotes
    """
    save_dir = save_dir or DATA_DIR / "preference"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading Stanford Human Preferences dataset...")
    
    dataset = load_dataset("stanfordnlp/SHP", split="train")
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Convert format
    train_data = []
    for item in dataset:
        prompt = item["history"]  # question/context
        
        # Determine which is chosen based on label
        if item["labels"] == 1:
            chosen = item["human_ref_A"]
            rejected = item["human_ref_B"]
        else:
            chosen = item["human_ref_B"]
            rejected = item["human_ref_A"]
        
        if prompt and chosen and rejected:
            train_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })
    
    # Split
    split_idx = int(len(train_data) * 0.95)
    train_split = train_data[:split_idx]
    val_split = train_data[split_idx:]
    
    # Save
    with open(save_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for item in train_split:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(save_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for item in val_split:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"SHP dataset download complete!")
    print(f"  Train set: {save_dir / 'train.jsonl'} ({len(train_split)} samples)")
    print(f"  Val set: {save_dir / 'val.jsonl'} ({len(val_split)} samples)")


def show_sample(data_path: Path, num: int = 3):
    """Display data samples"""
    print(f"\nData sample preview ({data_path}):")
    print("=" * 60)
    
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num:
                break
            item = json.loads(line)
            print(f"\n--- Sample {i + 1} ---")
            print(f"Prompt: {item['prompt'][:100]}...")
            print(f"Chosen: {item['chosen'][:100]}...")
            print(f"Rejected: {item['rejected'][:100]}...")
    
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download preference datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ultrafeedback",
        choices=["hh_rlhf", "ultrafeedback", "shp"],
        help="Select dataset"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to download"
    )
    parser.add_argument(
        "--show_sample",
        action="store_true",
        help="Show samples after download"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "hh_rlhf":
        download_hh_rlhf(num_samples=args.num_samples)
    elif args.dataset == "ultrafeedback":
        download_ultrafeedback(num_samples=args.num_samples)
    elif args.dataset == "shp":
        download_shp(num_samples=args.num_samples)
    
    if args.show_sample:
        show_sample(DATA_DIR / "preference" / "train.jsonl")