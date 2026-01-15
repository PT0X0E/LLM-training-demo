"""
DPO (Direct Preference Optimization) Training Script

DPO is a simplified alternative to RLHF:
- No need to train a separate reward model
- No complex PPO reinforcement learning required
- Directly optimize the policy model using preference data
"""

import sys
import math
import time
import json
import copy
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from torch.optim import AdamW

# Add project path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.transformer import GPT, GPTConfig
from src.data.tokenizer import LLMTokenizer, TOKENIZER_DIR
from src.data.preference_dataset import create_preference_dataloaders


DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs"


@dataclass
class DPOConfig:
    """DPO Training Configuration"""
    # Model
    sft_model_path: str = ""  # SFT model path (required)
    model_size: str = "small"  # If no SFT model
    
    # Data
    train_data: str = "preference/train.jsonl"
    val_data: str = "preference/val.jsonl"
    
    # DPO hyperparameters
    beta: float = 0.1  # KL divergence penalty coefficient, higher is more conservative
    label_smoothing: float = 0.0  # Label smoothing
    
    # Training parameters
    batch_size: int = 4
    max_seq_len: int = 512
    num_epochs: int = 1  # DPO usually only needs 1-3 epochs
    learning_rate: float = 5e-7  # DPO uses a very small learning rate
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 4
    
    # Logging and saving
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    output_dir: str = "dpo_run"
    
    # Device
    device: str = "auto"


class DPOTrainer:
    """
    DPO Trainer
    
    DPO Loss:
    L = -log(sigmoid(beta * (log_pi(chosen) - log_pi(rejected) - log_ref(chosen) + log_ref(rejected))))
    
    Where:
    - pi: current policy model
    - ref: reference model (frozen SFT model)
    - beta: KL penalty coefficient
    """
    
    def __init__(self, config: DPOConfig):
        self.config = config
        self.setup_device()
        self.setup_output_dir()
        self.setup_models()
        self.setup_data()
        self.setup_optimizer()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_logs = []
        self.val_logs = []
    
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
        
        with open(self.output_dir / "dpo_config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        
        print(f"Output directory: {self.output_dir}")
    
    def setup_models(self):
        """Set up policy and reference models"""
        self.tokenizer = LLMTokenizer(TOKENIZER_DIR)
        
        if self.config.sft_model_path:
            # Load SFT model
            sft_dir = OUTPUT_DIR / self.config.sft_model_path
            print(f"Loading SFT model: {sft_dir}")
            
            with open(sft_dir / "model_config.json", "r") as f:
                model_config_dict = json.load(f)
            model_config = GPTConfig(**model_config_dict)
            
            # Create policy model
            self.model = GPT(model_config)
            checkpoint = torch.load(sft_dir / "best_model.pt", map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
            # Create reference model (frozen)
            self.ref_model = GPT(model_config)
            self.ref_model.load_state_dict(checkpoint["model_state_dict"])
            
            print("SFT model loaded successfully!")
        else:
            # Create from scratch (for testing only)
            print("Warning: SFT model not specified, starting from scratch (for testing only)")
            config_map = {
                "tiny": GPTConfig.tiny,
                "small": GPTConfig.small,
                "medium": GPTConfig.medium,
                "large": GPTConfig.large,
            }
            model_config = config_map[self.config.model_size](self.tokenizer.vocab_size)
            model_config.max_seq_len = self.config.max_seq_len
            
            self.model = GPT(model_config)
            self.ref_model = GPT(model_config)
            self.ref_model.load_state_dict(self.model.state_dict())
        
        # Move to device
        self.model.to(self.device)
        self.ref_model.to(self.device)
        
        # Freeze reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.model.print_model_info()
        print(f"Reference model frozen")
    
    def setup_data(self):
        """Set up data"""
        train_path = DATA_DIR / self.config.train_data
        val_path = DATA_DIR / self.config.val_data
        
        if not train_path.exists():
            raise FileNotFoundError(
                f"Training data not found: {train_path}\n"
                f"Please run: python src/data/download_preference.py"
            )
        
        self.train_loader, self.val_loader = create_preference_dataloaders(
            train_path=train_path,
            val_path=val_path if val_path.exists() else None,
            tokenizer=self.tokenizer,
            max_seq_len=self.config.max_seq_len,
            batch_size=self.config.batch_size,
        )
        
        print(f"Number of training batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Number of validation batches: {len(self.val_loader)}")
    
    def setup_optimizer(self):
        """Set up optimizer"""
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
        """Cosine scheduler with warmup"""
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def get_log_probs(
        self,
        model: GPT,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sequence log probabilities
        
        Returns average log probability per sample
        """
        outputs = model(input_ids, attention_mask)
        logits = outputs["logits"]
        
        # Shift: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute log probabilities for each position
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Get target token log probabilities
        # [batch, seq_len-1]
        target_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Create mask (only compute for non-padding and non -100 positions)
        mask = (shift_labels != -100) & (shift_labels != self.tokenizer.pad_token_id)
        mask = mask.float()
        
        # Compute average log probability per sample
        sum_log_probs = (target_log_probs * mask).sum(dim=-1)
        count = mask.sum(dim=-1).clamp(min=1)
        avg_log_probs = sum_log_probs / count
        
        return avg_log_probs
    
    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute DPO loss
        
        L = -log(sigmoid(beta * (log_pi(chosen) - log_pi(rejected) - log_ref(chosen) + log_ref(rejected))))
        """
        # Compute log ratios
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        
        # Compute rewards (implicit reward)
        chosen_rewards = self.config.beta * chosen_logratios
        rejected_rewards = self.config.beta * rejected_logratios
        
        # DPO loss
        logits = chosen_rewards - rejected_rewards
        
        if self.config.label_smoothing > 0:
            # Loss with label smoothing
            losses = (
                -F.logsigmoid(logits) * (1 - self.config.label_smoothing)
                - F.logsigmoid(-logits) * self.config.label_smoothing
            )
        else:
            losses = -F.logsigmoid(logits)
        
        loss = losses.mean()
        
        # Compute accuracy (proportion where chosen reward > rejected reward)
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        # Compute margin (difference between chosen and rejected rewards)
        margin = (chosen_rewards - rejected_rewards).mean()
        
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "margin": margin.item(),
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
        }
        
        return loss, metrics
    
    def train_step(self, batch: dict) -> tuple[float, dict]:
        """Single training step"""
        self.model.train()
        
        # Move data to device
        chosen_input_ids = batch["chosen_input_ids"].to(self.device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
        chosen_labels = batch["chosen_labels"].to(self.device)
        
        rejected_input_ids = batch["rejected_input_ids"].to(self.device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
        rejected_labels = batch["rejected_labels"].to(self.device)
        
        # Compute policy model log probs
        policy_chosen_logps = self.get_log_probs(
            self.model, chosen_input_ids, chosen_attention_mask, chosen_labels
        )
        policy_rejected_logps = self.get_log_probs(
            self.model, rejected_input_ids, rejected_attention_mask, rejected_labels
        )
        
        # Compute reference model log probs
        with torch.no_grad():
            ref_chosen_logps = self.get_log_probs(
                self.ref_model, chosen_input_ids, chosen_attention_mask, chosen_labels
            )
            ref_rejected_logps = self.get_log_probs(
                self.ref_model, rejected_input_ids, rejected_attention_mask, rejected_labels
            )
        
        # Compute DPO loss
        loss, metrics = self.compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        )
        
        # Gradient accumulation
        scaled_loss = loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()
        
        return loss.item(), metrics
    
    def optimizer_step(self):
        """Optimizer step"""
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
    
    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluation"""
        if not self.val_loader:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_margin = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            chosen_input_ids = batch["chosen_input_ids"].to(self.device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
            chosen_labels = batch["chosen_labels"].to(self.device)
            
            rejected_input_ids = batch["rejected_input_ids"].to(self.device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
            rejected_labels = batch["rejected_labels"].to(self.device)
            
            policy_chosen_logps = self.get_log_probs(
                self.model, chosen_input_ids, chosen_attention_mask, chosen_labels
            )
            policy_rejected_logps = self.get_log_probs(
                self.model, rejected_input_ids, rejected_attention_mask, rejected_labels
            )
            
            ref_chosen_logps = self.get_log_probs(
                self.ref_model, chosen_input_ids, chosen_attention_mask, chosen_labels
            )
            ref_rejected_logps = self.get_log_probs(
                self.ref_model, rejected_input_ids, rejected_attention_mask, rejected_labels
            )
            
            loss, metrics = self.compute_dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
            )
            
            total_loss += metrics["loss"]
            total_accuracy += metrics["accuracy"]
            total_margin += metrics["margin"]
            num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "accuracy": total_accuracy / num_batches,
            "margin": total_margin / num_batches,
        }
    
    @torch.no_grad()
    def generate_response(self, prompt: str) -> str:
        """Generate response"""
        self.model.eval()
        
        full_prompt = f"User: {prompt}\nAssistant: "
        input_ids = torch.tensor(
            [self.tokenizer.encode(full_prompt, add_special_tokens=False)],
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
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
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
        
        with open(self.output_dir / "model_config.json", "w") as f:
            json.dump(asdict(self.model.config), f, indent=2)
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("Start DPO Training")
        print(f"Beta (KL penalty coefficient): {self.config.beta}")
        print("=" * 60)
        
        total_steps = (len(self.train_loader) // self.config.gradient_accumulation_steps) * self.config.num_epochs
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.config.num_epochs} ---")
            epoch_start = time.time()
            
            accumulated_metrics = {
                "loss": 0.0,
                "accuracy": 0.0,
                "margin": 0.0,
            }
            
            for step, batch in enumerate(self.train_loader):
                loss, metrics = self.train_step(batch)
                
                for k, v in metrics.items():
                    if k in accumulated_metrics:
                        accumulated_metrics[k] += v
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.global_step += 1
                    
                    # Compute average metrics
                    avg_metrics = {
                        k: v / self.config.gradient_accumulation_steps
                        for k, v in accumulated_metrics.items()
                    }
                    accumulated_metrics = {k: 0.0 for k in accumulated_metrics}
                    
                    # Logging
                    if self.global_step % self.config.log_interval == 0:
                        lr = self.optimizer.param_groups[0]["lr"]
                        elapsed = time.time() - start_time
                        
                        print(
                            f"Step {self.global_step}/{total_steps} | "
                            f"Loss: {avg_metrics['loss']:.4f} | "
                            f"Acc: {avg_metrics['accuracy']:.2%} | "
                            f"Margin: {avg_metrics['margin']:.4f} | "
                            f"LR: {lr:.2e}"
                        )
                        
                        self.train_logs.append({
                            "step": self.global_step,
                            **avg_metrics,
                        })
                    
                    # Evaluation
                    if self.global_step % self.config.eval_interval == 0:
                        val_metrics = self.evaluate()
                        
                        if val_metrics:
                            print(
                                f"\n>>> Validation | "
                                f"Loss: {val_metrics['loss']:.4f} | "
                                f"Acc: {val_metrics['accuracy']:.2%} | "
                                f"Margin: {val_metrics['margin']:.4f}"
                            )
                            
                            self.val_logs.append({
                                "step": self.global_step,
                                **val_metrics,
                            })
                            
                            if val_metrics["loss"] < self.best_val_loss:
                                self.best_val_loss = val_metrics["loss"]
                                self.save_checkpoint("best_model")
                        
                        # Test generation
                        test_prompts = [
                            "What is the capital of France?",
                            "Explain machine learning in simple terms.",
                        ]
                        
                        print("\n>>> Generation test:")
                        for prompt in test_prompts:
                            response = self.generate_response(prompt)
                            print(f"  Q: {prompt}")
                            print(f"  A: {response[:150]}...")
                        print()
                    
                    # Periodic saving
                    if self.global_step % self.config.save_interval == 0:
                        self.save_checkpoint(f"step_{self.global_step}")
            
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch + 1} finished | Time: {epoch_time:.1f}s")
        
        # Training finished
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"DPO Training finished! Total time: {total_time / 60:.1f} minutes")
        print("=" * 60)
        
        self.save_checkpoint("final_model")
        
        # Save logs
        logs = {
            "train_logs": self.train_logs,
            "val_logs": self.val_logs,
            "best_val_loss": self.best_val_loss,
            "total_steps": self.global_step,
            "total_time": total_time,
        }
        with open(self.output_dir / "training_logs.json", "w") as f:
            json.dump(logs, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="DPO Preference Optimization Training")
    
    # Model
    parser.add_argument("--sft_model_path", type=str, default="",
                        help="SFT model path, e.g. sft_run")
    parser.add_argument("--model_size", type=str, default="small",
                        choices=["tiny", "small", "medium", "large"])
    
    # Data
    parser.add_argument("--train_data", type=str, default="preference/train.jsonl")
    parser.add_argument("--val_data", type=str, default="preference/val.jsonl")
    
    # DPO hyperparameters
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty coefficient, higher is more conservative")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="dpo_run")
    
    # Device
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    config = DPOConfig(**vars(args))
    trainer = DPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()