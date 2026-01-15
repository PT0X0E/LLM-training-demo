"""
Pretraining script
Implements GPT model pretraining workflow
"""

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
from torch.amp import autocast, GradScaler  # Updated for deprecation

# Add project path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.transformer import GPT, GPTConfig
from src.data.tokenizer import LLMTokenizer, TOKENIZER_DIR
from src.data.pretrain_dataset import create_dataloaders


DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs"


@dataclass
class TrainConfig:
    """Training configuration"""
    # Data
    train_data: str = "wikitext-2/train.txt"
    val_data: str = "wikitext-2/validation.txt"
    
    # Model size
    model_size: str = "small"  # tiny, small, medium, large
    
    # Training parameters
    batch_size: int = 8
    max_seq_len: int = 256
    num_epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    grad_clip: float = 1.0
    
    # Optimization
    use_amp: bool = True  # Mixed precision training
    gradient_accumulation_steps: int = 1
    
    # Logging and saving
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    output_dir: str = "pretrain_run"
    
    # Device
    device: str = "auto"  # auto, cpu, cuda, mps


class Trainer:
    """Pretrainer"""
    
    def __init__(self, config: TrainConfig):
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
        
        # MPS does not support AMP
        if self.device.type == "mps":
            self.config.use_amp = False
    
    def setup_output_dir(self):
        """Set output directory"""
        self.output_dir = OUTPUT_DIR / self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "train_config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        
        print(f"Output directory: {self.output_dir}")
    
    def setup_model(self):
        """Set up model"""
        # Load tokenizer
        self.tokenizer = LLMTokenizer(TOKENIZER_DIR)
        
        # Create model config
        config_map = {
            "tiny": GPTConfig.tiny,
            "small": GPTConfig.small,
            "medium": GPTConfig.medium,
            "large": GPTConfig.large,
        }
        
        model_config = config_map[self.config.model_size](self.tokenizer.vocab_size)
        model_config.max_seq_len = self.config.max_seq_len
        
        # Create model
        self.model = GPT(model_config)
        self.model.to(self.device)
        self.model.print_model_info()
        
        # Save model config
        with open(self.output_dir / "model_config.json", "w") as f:
            json.dump(asdict(model_config), f, indent=2)
    
    def setup_data(self):
        """Set up data"""
        train_path = DATA_DIR / self.config.train_data
        val_path = DATA_DIR / self.config.val_data
        
        self.train_loader, self.val_loader = create_dataloaders(
            train_path=train_path,
            val_path=val_path,
            tokenizer=self.tokenizer,
            max_seq_len=self.config.max_seq_len,
            batch_size=self.config.batch_size,
        )
        
        print(f"Number of training batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Number of validation batches: {len(self.val_loader)}")
    
    def setup_optimizer(self):
        """Set up optimizer and scheduler"""
        # Separate parameters with and without weight decay
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
        
        # Scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.1,
        )
        
        # Mixed precision
        self.scaler = GradScaler("cuda") if self.config.use_amp and self.device.type == "cuda" else None
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]["lr"]
    
    def train_step(self, batch: dict) -> float:
        """Single training step"""
        self.model.train()
        
        # Move data to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Forward pass
        if self.config.use_amp and self.scaler:
            with autocast("cuda"):
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs["loss"] / self.config.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs["loss"] / self.config.gradient_accumulation_steps
            loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def optimizer_step(self):
        """Optimizer step"""
        if self.config.grad_clip > 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad()
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate model"""
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
    def generate_sample(self, prompt: str = "The") -> str:
        """Generate sample text"""
        self.model.eval()
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt, add_special_tokens=False)],
            device=self.device
        )
        
        generated = self.model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
        )
        
        return self.tokenizer.decode(generated[0].tolist())
    
    def save_checkpoint(self, name: str = "checkpoint"):
        """Save checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        path = self.output_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Path):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        print(f"Checkpoint loaded: {path}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("Start pretraining")
        print("=" * 60)
        
        total_steps = len(self.train_loader) * self.config.num_epochs
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.config.num_epochs} ---")
            epoch_loss = 0.0
            epoch_start = time.time()
            
            for step, batch in enumerate(self.train_loader):
                # Train one step
                loss = self.train_step(batch)
                epoch_loss += loss
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.log_interval == 0:
                        avg_loss = epoch_loss / (step + 1)
                        lr = self.get_lr()
                        elapsed = time.time() - start_time
                        steps_per_sec = self.global_step / elapsed
                        
                        print(
                            f"Step {self.global_step}/{total_steps} | "
                            f"Loss: {loss:.4f} | "
                            f"Avg Loss: {avg_loss:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Speed: {steps_per_sec:.2f} steps/s"
                        )
                        
                        self.train_losses.append({
                            "step": self.global_step,
                            "loss": loss,
                        })
                    
                    # Evaluation
                    if self.global_step % self.config.eval_interval == 0:
                        val_loss = self.evaluate()
                        ppl = math.exp(val_loss) if val_loss < 10 else float("inf")
                        
                        print(f"\n>>> Validation | Loss: {val_loss:.4f} | PPL: {ppl:.2f}")
                        
                        # Generate sample
                        sample = self.generate_sample("The")
                        print(f">>> Sample generated: {sample[:100]}...")
                        print()
                        
                        self.val_losses.append({
                            "step": self.global_step,
                            "loss": val_loss,
                            "ppl": ppl,
                        })
                        
                        # Save best model
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint("best_model")
                    
                    # Periodic saving
                    if self.global_step % self.config.save_interval == 0:
                        self.save_checkpoint(f"step_{self.global_step}")
            
            # Epoch end statistics
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            print(f"\nEpoch {epoch + 1} finished | Avg Loss: {avg_epoch_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # Training finished
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Pretraining finished! Total time: {total_time / 60:.1f} minutes")
        print("=" * 60)
        
        # Save final model
        self.save_checkpoint("final_model")
        
        # Save training logs
        logs = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "total_steps": self.global_step,
            "total_time": total_time,
        }
        with open(self.output_dir / "training_logs.json", "w") as f:
            json.dump(logs, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="GPT Pretraining")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, default="wikitext-2/train.txt")
    parser.add_argument("--val_data", type=str, default="wikitext-2/validation.txt")
    
    # Model arguments
    parser.add_argument("--model_size", type=str, default="small",
                        choices=["tiny", "small", "medium", "large"])
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # Optimization
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="pretrain_run")
    
    # Device
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainConfig(**vars(args))
    
    # Start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()