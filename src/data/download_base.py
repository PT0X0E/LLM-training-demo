"""
Data download module
Supports downloading various small high-quality datasets for LLM training
"""

from datasets import load_dataset
from pathlib import Path
import json


DATA_DIR = Path(__file__).parent.parent.parent / "data"


def download_wikitext2(save_dir: Path = None) -> dict:
    """
    Download WikiText-2 dataset
    - About 2MB, contains Wikipedia articles
    - Suitable for quick experiments and learning
    """
    save_dir = save_dir or DATA_DIR / "wikitext-2"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Save as text files for easy viewing
    for split in ["train", "validation", "test"]:
        texts = [item["text"] for item in dataset[split] if item["text"].strip()]
        output_path = save_dir / f"{split}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(texts))
        print(f"  Saved {split}: {len(texts)} texts -> {output_path}")
    
    # Save dataset info
    info = {
        "name": "wikitext-2-raw-v1",
        "splits": {
            split: len(dataset[split]) for split in dataset.keys()
        }
    }
    with open(save_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"WikiText-2 download complete! Saved to: {save_dir}")
    return dataset


def download_tiny_shakespeare(save_dir: Path = None) -> dict:
    """
    Download Tiny Shakespeare dataset
    - About 1MB, complete works of Shakespeare
    - Very small, suitable for quick debugging
    """
    save_dir = save_dir or DATA_DIR / "tiny_shakespeare"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading Tiny Shakespeare dataset...")
    dataset = load_dataset("tiny_shakespeare")
    
    for split in dataset.keys():
        texts = [item["text"] for item in dataset[split]]
        output_path = save_dir / f"{split}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(texts))
        print(f"  Saved {split}: {len(texts)} texts -> {output_path}")
    
    print(f"Tiny Shakespeare download complete! Saved to: {save_dir}")
    return dataset


def download_chinese_wiki_simple(save_dir: Path = None, num_samples: int = 10000) -> dict:
    """
    Download Chinese Wikipedia subset
    - Controllable sample count
    - Suitable for Chinese LLM training
    """
    save_dir = save_dir or DATA_DIR / "chinese_wiki"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Chinese Wikipedia (first {num_samples} entries)...")
    dataset = load_dataset(
        "wikipedia", 
        "20220301.zh",
        split=f"train[:{num_samples}]",
        trust_remote_code=True
    )
    
    # Save texts
    texts = [item["text"] for item in dataset if item["text"].strip()]
    output_path = save_dir / "train.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(texts))
    
    print(f"Chinese Wikipedia download complete! {len(texts)} texts -> {output_path}")
    return dataset


def download_all():
    """Download all recommended datasets"""
    print("=" * 50)
    print("Starting download of all datasets")
    print("=" * 50)
    
    download_wikitext2()
    print()
    download_tiny_shakespeare()
    print()
    
    print("=" * 50)
    print("All datasets downloaded!")
    print(f"Data directory: {DATA_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download LLM training datasets")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="all",
        choices=["wikitext2", "shakespeare", "chinese_wiki", "all"],
        help="Select dataset to download"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to download for Chinese Wikipedia"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "wikitext2":
        download_wikitext2()
    elif args.dataset == "shakespeare":
        download_tiny_shakespeare()
    elif args.dataset == "chinese_wiki":
        download_chinese_wiki_simple(num_samples=args.num_samples)
    else:
        download_all()