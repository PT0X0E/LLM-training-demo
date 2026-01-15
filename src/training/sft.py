"""
SFT instruction fine-tuning training script
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.transformer import GPT, GPTConfig
from src.data.tokenizer import LLMTokenizer, TOKENIZER_DIR
from src.data.sft_dataset import create_sft_dataloaders


DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs"


@dataclass
class SFTConfig:
    """SFT training configuration"""
    # Model
    pretrained_path: str = ""  # Pretrained model path, empty for training from scratch
    model_size: str = "small"  # If no pretrained model, use this config
    
    # Data
    train_data: str = "sft/train.jsonl"
    val_data: str = "sft/val.jsonl"
    prompt_template: str = "simple"  # alpaca, chatml, simple
    
    # Training parameters
    batch_size: int = 4
    max_seq_len: int = 512
    num_epochs: int = 3
    learning_rate: float = 2e-5  # SFT usually uses a smaller learning rate
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 4
    
    # Logging and saving
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    output_dir: str = "sft_run"
    
    # Device
    device: str = "auto"


class SFTTrainer:
    """SFT instruction fine-tuning trainer"""
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.setup_device()
        self.setup_output_dir()
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []
    
    def setup_device(self):
        """Set device"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)
        
        print(f"Using device: {self.device}")
    
    def setup_output_dir(self):
        """Set output directory"""
        self.output_dir = OUTPUT_DIR / self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_dir / "sft_config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        
        print(f"Output directory: {self.output_dir}")
    
    def setup_model(self):
        """Set model"""
        self.tokenizer = LLMTokenizer(TOKENIZER_DIR)
        
        if self.config.pretrained_path:
            # Load pretrained model
            pretrained_dir = OUTPUT_DIR / self.config.pretrained_path
            print(f"Loading pretrained model: {pretrained_dir}")
            
            # Load model config
            with open(pretrained_dir / "model_config.json", "r") as f:
                model_config_dict = json.load(f)
            model_config = GPTConfig(**model_config_dict)
            
            # Create model
            self.model = GPT(model_config)
            
            # Load weights
            checkpoint = torch.load(
                pretrained_dir / "best_model.pt",
                map_location=self.device
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print("Pretrained weights loaded successfully!")
        else:
            # Create model from scratch
            print("No pretrained model specified, training from scratch")
            config_map = {
                "tiny": GPTConfig.tiny,
                "small": GPTConfig.small,
                "medium": GPTConfig.medium,
                "large": GPTConfig.large,
            }
            model_config = config_map[self.config.model_size](self.tokenizer.vocab_size)
            model_config.max_seq_len = self.config.max_seq_len
            self.model = GPT(model_config)
        
        self.model.to(self.device)
        self.model.print_model_info()
    
    def setup_data(self):
        """Set data"""
        train_path = DATA_DIR / self.config.train_data
        val_path = DATA_DIR / self.config.val_data
        
        if not train_path.exists():
            raise FileNotFoundError(
                f"Training data not found: {train_path}\n"
                f"Please run: python -c \"from src.training.sft import create_sample_data; create_sample_data()\""
            )
        
        self.train_loader, self.val_loader = create_sft_dataloaders(
            train_path=train_path,
            val_path=val_path if val_path.exists() else None,
            tokenizer=self.tokenizer,
            max_seq_len=self.config.max_seq_len,
            batch_size=self.config.batch_size,
            prompt_template=self.config.prompt_template,
        )
        
        print(f"Number of training batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Number of validation batches: {len(self.val_loader)}")
    
    def setup_optimizer(self):
        """Set optimizer"""
        # Separate parameters
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "weight" in name and "norm" not in name:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        self.optimizer = AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
        )
        
        # Learning rate scheduler
        total_steps = (len(self.train_loader) // self.config.gradient_accumulation_steps) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        self.scheduler = self.get_scheduler(total_steps, warmup_steps)
        
        print(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    def get_scheduler(self, total_steps: int, warmup_steps: int):
        """Get cosine scheduler with warmup"""
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
    
    def train_step(self, batch: dict) -> float:
        """Single training step"""
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        outputs = self.model(input_ids, attention_mask, labels)
        loss = outputs["loss"] / self.config.gradient_accumulation_steps
        loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def optimizer_step(self):
        """Optimizer step"""
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluation"""
        if not self.val_loader:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model(input_ids, attention_mask, labels)
            total_loss += outputs["loss"].item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def generate_response(self, instruction: str, input_text: str = "") -> str:
        """Generate response"""
        self.model.eval()
        
        # Build prompt
        if self.config.prompt_template == "alpaca":
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
        else:
            if input_text:
                prompt = f"User: {instruction}\n{input_text}\nAssistant: "
            else:
                prompt = f"User: {instruction}\nAssistant: "
        
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt, add_special_tokens=False)],
            device=self.device
        )
        
        generated = self.model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
        )
        
        response = self.tokenizer.decode(generated[0].tolist())
        # Only return the generated part
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        elif "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response
    
    def save_checkpoint(self, name: str = "checkpoint"):
        """Save checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        
        path = self.output_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
        
        # Also save model config
        with open(self.output_dir / "model_config.json", "w") as f:
            json.dump(asdict(self.model.config), f, indent=2)
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("Start SFT instruction fine-tuning")
        print("=" * 60)
        
        total_steps = (len(self.train_loader) // self.config.gradient_accumulation_steps) * self.config.num_epochs
        start_time = time.time()
        accumulation_loss = 0.0
        
        for epoch in range(self.config.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.config.num_epochs} ---")
            epoch_loss = 0.0
            epoch_start = time.time()
            
            for step, batch in enumerate(self.train_loader):
                loss = self.train_step(batch)
                accumulation_loss += loss
                epoch_loss += loss
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.global_step += 1
                    
                    avg_loss = accumulation_loss / self.config.gradient_accumulation_steps
                    accumulation_loss = 0.0
                    
                    # Logging
                    if self.global_step % self.config.log_interval == 0:
                        lr = self.get_lr()
                        elapsed = time.time() - start_time
                        steps_per_sec = self.global_step / elapsed
                        
                        print(
                            f"Step {self.global_step}/{total_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Speed: {steps_per_sec:.2f} steps/s"
                        )
                        
                        self.train_losses.append({
                            "step": self.global_step,
                            "loss": avg_loss,
                        })
                    
                    # Evaluation
                    if self.global_step % self.config.eval_interval == 0:
                        val_loss = self.evaluate()
                        
                        print(f"\n>>> Validation Loss: {val_loss:.4f}")
                        
                        # Test generation
                        test_instructions = [
                            "What is the capital of France?",
                            "Explain what machine learning is in simple terms.",
                        ]
                        
                        for instr in test_instructions:
                            response = self.generate_response(instr)
                            print(f">>> Q: {instr}")
                            print(f">>> A: {response[:150]}...")
                        print()
                        
                        self.val_losses.append({
                            "step": self.global_step,
                            "loss": val_loss,
                        })
                        
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint("best_model")
                    
                    # Periodic saving
                    if self.global_step % self.config.save_interval == 0:
                        self.save_checkpoint(f"step_{self.global_step}")
            
            # End of epoch
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            print(f"\nEpoch {epoch + 1} finished | Average Loss: {avg_epoch_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # Training finished
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"SFT training finished! Total time: {total_time / 60:.1f} minutes")
        print("=" * 60)
        
        self.save_checkpoint("final_model")
        
        # Save logs
        logs = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "total_steps": self.global_step,
            "total_time": total_time,
        }
        with open(self.output_dir / "training_logs.json", "w") as f:
            json.dump(logs, f, indent=2)


def create_sample_data():
    """Create sample SFT data"""
    sft_dir = DATA_DIR / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample training data
    train_data = [
        {"instruction": "What is the capital of France?", "input": "", "output": "The capital of France is Paris."},
        {"instruction": "Translate the following to French.", "input": "Hello, how are you?", "output": "Bonjour, comment allez-vous?"},
        {"instruction": "Summarize the following text.", "input": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.", "output": "Machine learning is an AI subset that allows systems to learn from data."},
        {"instruction": "What is 2 + 2?", "input": "", "output": "2 + 2 equals 4."},
        {"instruction": "Write a short poem about the moon.", "input": "", "output": "The moon shines bright,\nA silver light,\nGuiding us through the night."},
        {"instruction": "Explain photosynthesis.", "input": "", "output": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen."},
        {"instruction": "What are the primary colors?", "input": "", "output": "The primary colors are red, blue, and yellow."},
        {"instruction": "Convert Celsius to Fahrenheit.", "input": "25 degrees Celsius", "output": "25 degrees Celsius is equal to 77 degrees Fahrenheit."},
        {"instruction": "List three programming languages.", "input": "", "output": "Three programming languages are Python, JavaScript, and Java."},
        {"instruction": "What is the largest planet in our solar system?", "input": "", "output": "Jupiter is the largest planet in our solar system."},
        {"instruction": "Define artificial intelligence.", "input": "", "output": "Artificial intelligence is the simulation of human intelligence by machines."},
        {"instruction": "What is the speed of light?", "input": "", "output": "The speed of light is approximately 299,792 kilometers per second."},
        {"instruction": "Name three fruits.", "input": "", "output": "Three fruits are apple, banana, and orange."},
        {"instruction": "What is the boiling point of water?", "input": "", "output": "The boiling point of water is 100 degrees Celsius or 212 degrees Fahrenheit."},
        {"instruction": "Explain gravity.", "input": "", "output": "Gravity is a force that attracts objects with mass toward each other."},
        {"instruction": "What is DNA?", "input": "", "output": "DNA is deoxyribonucleic acid, the molecule that carries genetic information in living organisms."},
        {"instruction": "Write a greeting.", "input": "", "output": "Hello! Welcome, it's nice to meet you."},
        {"instruction": "What causes rain?", "input": "", "output": "Rain is caused by water vapor in clouds condensing into droplets that fall to Earth."},
        {"instruction": "Name the continents.", "input": "", "output": "The seven continents are Africa, Antarctica, Asia, Australia, Europe, North America, and South America."},
        {"instruction": "What is an algorithm?", "input": "", "output": "An algorithm is a step-by-step procedure for solving a problem or completing a task."},
    ]
    
    # Validation data
    val_data = [
        {"instruction": "What is the capital of Japan?", "input": "", "output": "The capital of Japan is Tokyo."},
        {"instruction": "Explain what a neural network is.", "input": "", "output": "A neural network is a computing system inspired by biological neural networks in the brain."},
        {"instruction": "What is 10 times 5?", "input": "", "output": "10 times 5 equals 50."},
        {"instruction": "Name the planets in our solar system.", "input": "", "output": "The planets are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."},
        {"instruction": "What is the chemical formula for water?", "input": "", "output": "The chemical formula for water is H2O."},
    ]
    
    # Write to file
    with open(sft_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(sft_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Sample data created:")
    print(f"  Training set: {sft_dir / 'train.jsonl'} ({len(train_data)} samples)")
    print(f"  Validation set: {sft_dir / 'val.jsonl'} ({len(val_data)} samples)")


def main():
    parser = argparse.ArgumentParser(description="SFT instruction fine-tuning")
    
    # Model
    parser.add_argument("--pretrained_path", type=str, default="",
                        help="Pretrained model path, e.g. pretrain_run")
    parser.add_argument("--model_size", type=str, default="small",
                        choices=["tiny", "small", "medium", "large"])
    
    # Data
    parser.add_argument("--train_data", type=str, default="sft/train.jsonl")
    parser.add_argument("--val_data", type=str, default="sft/val.jsonl")
    parser.add_argument("--prompt_template", type=str, default="simple",
                        choices=["alpaca", "chatml", "simple"])
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="sft_run")
    
    # Device
    parser.add_argument("--device", type=str, default="auto")
    
    # Special command
    parser.add_argument("--create_sample_data", action="store_true",
                        help="Create sample data")
    
    args = parser.parse_args()
    
    if args.create_sample_data:
        create_sample_data()
        return
    
    config = SFTConfig(**{k: v for k, v in vars(args).items() if k != "create_sample_data"})
    trainer = SFTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()