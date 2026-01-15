"""
Tokenizer Module
Supports training BPE Tokenizer from scratch or loading pretrained Tokenizer
"""

from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast
import json


DATA_DIR = Path(__file__).parent.parent.parent / "data"
TOKENIZER_DIR = DATA_DIR / "tokenizer"


class BPETokenizerTrainer:
    """
    Train a BPE (Byte Pair Encoding) Tokenizer from scratch
    This is the most commonly used tokenization method for modern LLMs
    """
    
    def __init__(self, vocab_size: int = 8000):
        """
        Args:
            vocab_size: Vocabulary size, 8000-16000 recommended for small models
        """
        self.vocab_size = vocab_size
        self.tokenizer = None
        
        # Special tokens
        self.special_tokens = [
            "<pad>",    # Padding
            "<unk>",    # Unknown token
            "<bos>",    # Beginning of sentence
            "<eos>",    # End of sentence
        ]
    
    def train(self, files: list[str], save_dir: Path = None) -> Tokenizer:
        """
        Train BPE Tokenizer on given text files
        
        Args:
            files: List of training file paths
            save_dir: Save directory
        """
        save_dir = save_dir or TOKENIZER_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting BPE Tokenizer training...")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Training files: {files}")
        
        # 1. Initialize BPE model
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        
        # 2. Set pre-tokenizer (split by whitespace and punctuation)
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # 3. Set trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        
        # 4. Train
        tokenizer.train(files, trainer)
        
        # 5. Set post-processor (add BOS/EOS)
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[
                ("<bos>", tokenizer.token_to_id("<bos>")),
                ("<eos>", tokenizer.token_to_id("<eos>")),
            ]
        )
        
        # 6. Set decoder
        tokenizer.decoder = decoders.ByteLevel()
        
        # 7. Save
        tokenizer_path = save_dir / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        
        # Save configuration
        config = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "model_type": "BPE"
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        self.tokenizer = tokenizer
        print(f"Tokenizer training complete! Saved to: {save_dir}")
        print(f"  Actual vocabulary size: {tokenizer.get_vocab_size()}")
        
        return tokenizer
    
    def train_on_wikitext2(self) -> Tokenizer:
        """Train on WikiText-2 dataset"""
        train_file = DATA_DIR / "wikitext-2" / "train.txt"
        if not train_file.exists():
            raise FileNotFoundError(f"Please download the dataset first: {train_file}")
        return self.train([str(train_file)])


class LLMTokenizer:
    """
    LLM Tokenizer wrapper class
    Provides unified interface, supports training and loading
    """
    
    def __init__(self, tokenizer_path: Path = None):
        """
        Args:
            tokenizer_path: Tokenizer file path, call train() or load() if None
        """
        self.tokenizer = None
        self.vocab_size = 0
        
        if tokenizer_path:
            self.load(tokenizer_path)
    
    def train(self, files: list[str], vocab_size: int = 8000, save_dir: Path = None):
        """Train a new Tokenizer"""
        trainer = BPETokenizerTrainer(vocab_size=vocab_size)
        base_tokenizer = trainer.train(files, save_dir or TOKENIZER_DIR)
        
        # Convert to HuggingFace format for easier use
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=base_tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
        )
        self.vocab_size = self.tokenizer.vocab_size
        
        # Save in HuggingFace format
        save_path = save_dir or TOKENIZER_DIR
        self.tokenizer.save_pretrained(save_path)
        print(f"HuggingFace format saved to: {save_path}")
    
    def load(self, tokenizer_path: Path):
        """Load a trained Tokenizer"""
        tokenizer_path = Path(tokenizer_path)
        
        if (tokenizer_path / "tokenizer_config.json").exists():
            # HuggingFace format
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
        elif (tokenizer_path / "tokenizer.json").exists():
            # Original tokenizers format
            base_tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=base_tokenizer,
                unk_token="<unk>",
                pad_token="<pad>",
                bos_token="<bos>",
                eos_token="<eos>",
            )
        else:
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        
        self.vocab_size = self.tokenizer.vocab_size
        print(f"Tokenizer loaded, vocabulary size: {self.vocab_size}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs"""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_encode(
        self, 
        texts: list[str], 
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> dict:
        """Batch encode, returns format ready for model input"""
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
    
    def get_vocab(self) -> dict:
        """Get vocabulary"""
        return self.tokenizer.get_vocab()
    
    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id
    
    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id
    
    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id


def demo():
    """Demonstrate Tokenizer training and usage"""
    print("=" * 60)
    print("Tokenizer Training Demo")
    print("=" * 60)
    
    # 1. Train
    tokenizer = LLMTokenizer()
    train_file = DATA_DIR / "wikitext-2" / "train.txt"
    
    if not train_file.exists():
        print(f"Error: Please run download.py first to download data")
        return
    
    tokenizer.train(
        files=[str(train_file)],
        vocab_size=8000,
        save_dir=TOKENIZER_DIR
    )
    
    # 2. Test encoding and decoding
    print("\n" + "=" * 60)
    print("Encoding/Decoding Test")
    print("=" * 60)
    
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
    ]
    
    for text in test_texts:
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)
        tokens = tokenizer.tokenizer.convert_ids_to_tokens(token_ids)
        
        print(f"\nOriginal: {text}")
        print(f"Token IDs: {token_ids}")
        print(f"Tokens: {tokens}")
        print(f"Decoded: {decoded}")
    
    # 3. Batch encoding
    print("\n" + "=" * 60)
    print("Batch Encoding Test")
    print("=" * 60)
    
    batch = tokenizer.batch_encode(test_texts, max_length=32)
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
    
    # 4. Vocabulary info
    print("\n" + "=" * 60)
    print("Vocabulary Info")
    print("=" * 60)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    print(f"BOS token ID: {tokenizer.bos_token_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or test Tokenizer")
    parser.add_argument(
        "--action",
        type=str,
        default="demo",
        choices=["train", "demo"],
        help="Action to perform"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=8000,
        help="Vocabulary size"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Training data file path"
    )
    
    args = parser.parse_args()
    
    if args.action == "demo":
        demo()
    elif args.action == "train":
        data_file = args.data_file or str(DATA_DIR / "wikitext-2" / "train.txt")
        tokenizer = LLMTokenizer()
        tokenizer.train(
            files=[data_file],
            vocab_size=args.vocab_size
        )