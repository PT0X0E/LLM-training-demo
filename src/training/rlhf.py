"""
RLHF (Reinforcement Learning from Human Feedback) training script

RLHF consists of three stages:
1. Train the Reward Model
2. Optimize the policy model using PPO
3. Apply KL penalty to prevent model drift

Compared to DPO, RLHF is more complex but more flexible
"""

import sys
import math
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Add project path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.transformer import GPT, GPTConfig
from src.data.tokenizer import LLMTokenizer, TOKENIZER_DIR
from src.data.preference_dataset import PreferenceDataset


DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs"


# =============================================================================
# Reward Model
# =============================================================================

class RewardModel(nn.Module):
    """
    Reward Model
    
    Based on GPT architecture, outputs a scalar reward value instead of token probabilities
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Reuse GPT backbone
        self.backbone = GPT(config)
        
        # Remove lm_head, replace with reward head
        self.backbone.lm_head = nn.Identity()
        
        # Reward head: maps hidden states to a scalar
        self.reward_head = nn.Linear(config.d_model, 1, bias=False)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            rewards: [batch_size] reward value for each sequence
        """
        # Get hidden states
        x = self.backbone.token_embedding(input_ids)
        x = self.backbone.dropout(x)
        
        for layer in self.backbone.layers:
            x = layer(x, attention_mask)
        
        x = self.backbone.final_norm(x)  # [batch, seq_len, d_model]
        
        # Take the last non-padding token's hidden state
        if attention_mask is not None:
            # Find the last valid position for each sequence
            seq_lengths = attention_mask.sum(dim=-1) - 1  # [batch]
            batch_indices = torch.arange(x.size(0), device=x.device)
            last_hidden = x[batch_indices, seq_lengths]  # [batch, d_model]
        else:
            last_hidden = x[:, -1, :]  # [batch, d_model]
        
        # Compute reward
        rewards = self.reward_head(last_hidden).squeeze(-1)  # [batch]
        
        return rewards
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Reward Model Trainer
# =============================================================================

@dataclass
class RewardModelConfig:
    """Reward model training config"""
    # Model
    sft_model_path: str = ""
    model_size: str = "small"
    
    # Data
    train_data: str = "preference/train.jsonl"
    val_data: str = "preference/val.jsonl"
    
    # Training parameters
    batch_size: int = 4
    max_seq_len: int = 512
    num_epochs: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    output_dir: str = "reward_model_run"
    
    # Device
    device: str = "auto"


class RewardModelTrainer:
    """
    Reward Model Trainer
    
    Uses pairwise ranking loss:
    L = -log(sigmoid(r_chosen - r_rejected))
    """
    
    def __init__(self, config: RewardModelConfig):
        self.config = config
        self.setup_device()
        self.setup_output_dir()
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        
        self.global_step = 0
        self.best_accuracy = 0.0
        self.train_logs = []
        self.val_logs = []
    
    def setup_device(self):
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
        self.output_dir = OUTPUT_DIR / self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_dir / "reward_config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        print(f"Output directory: {self.output_dir}")
    
    def setup_model(self):
        self.tokenizer = LLMTokenizer(TOKENIZER_DIR)
        
        if self.config.sft_model_path:
            sft_dir = OUTPUT_DIR / self.config.sft_model_path
            print(f"Initializing reward model from SFT model: {sft_dir}")
            
            with open(sft_dir / "model_config.json", "r") as f:
                model_config_dict = json.load(f)
            model_config = GPTConfig(**model_config_dict)
            
            self.model = RewardModel(model_config)
            
            # Load SFT weights into backbone
            checkpoint = torch.load(sft_dir / "best_model.pt", map_location=self.device)
            # Filter out lm_head weights
            sft_state_dict = {
                k: v for k, v in checkpoint["model_state_dict"].items()
                if "lm_head" not in k
            }
            self.model.backbone.load_state_dict(sft_state_dict, strict=False)
            print("SFT weights loaded successfully!")
        else:
            print("Creating reward model from scratch")
            config_map = {
                "tiny": GPTConfig.tiny,
                "small": GPTConfig.small,
                "medium": GPTConfig.medium,
                "large": GPTConfig.large,
            }
            model_config = config_map[self.config.model_size](self.tokenizer.vocab_size)
            model_config.max_seq_len = self.config.max_seq_len
            self.model = RewardModel(model_config)
        
        self.model.to(self.device)
        print(f"Reward model parameter count: {self.model.count_parameters():,}")
        
        # Save config
        with open(self.output_dir / "model_config.json", "w") as f:
            json.dump(asdict(model_config), f, indent=2)
    
    def setup_data(self):
        train_path = DATA_DIR / self.config.train_data
        val_path = DATA_DIR / self.config.val_data
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        train_dataset = PreferenceDataset(train_path, self.tokenizer, self.config.max_seq_len)
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True, pin_memory=True
        )
        
        self.val_loader = None
        if val_path.exists():
            val_dataset = PreferenceDataset(val_path, self.tokenizer, self.config.max_seq_len)
            self.val_loader = DataLoader(
                val_dataset, batch_size=self.config.batch_size, shuffle=False, pin_memory=True
            )
        
        print(f"Number of training batches: {len(self.train_loader)}")
    
    def setup_optimizer(self):
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        total_steps = len(self.train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch: dict) -> Tuple[float, float]:
        self.model.train()
        
        # Get rewards for chosen and rejected
        chosen_rewards = self.model(
            batch["chosen_input_ids"].to(self.device),
            batch["chosen_attention_mask"].to(self.device),
        )
        rejected_rewards = self.model(
            batch["rejected_input_ids"].to(self.device),
            batch["rejected_attention_mask"].to(self.device),
        )
        
        # Pairwise ranking loss
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        # Accuracy
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        loss.backward()
        
        return loss.item(), accuracy.item()
    
    @torch.no_grad()
    def evaluate(self) -> dict:
        if not self.val_loader:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            chosen_rewards = self.model(
                batch["chosen_input_ids"].to(self.device),
                batch["chosen_attention_mask"].to(self.device),
            )
            rejected_rewards = self.model(
                batch["rejected_input_ids"].to(self.device),
                batch["rejected_attention_mask"].to(self.device),
            )
            
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "accuracy": total_accuracy / num_batches,
        }
    
    def save_checkpoint(self, name: str):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_accuracy": self.best_accuracy,
        }
        torch.save(checkpoint, self.output_dir / f"{name}.pt")
        print(f"Checkpoint saved: {self.output_dir / name}.pt")
    
    def train(self):
        print("\n" + "=" * 60)
        print("Start training reward model")
        print("=" * 60)
        
        total_steps = len(self.train_loader) * self.config.num_epochs
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.config.num_epochs} ---")
            
            for step, batch in enumerate(self.train_loader):
                loss, accuracy = self.train_step(batch)
                
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                if self.global_step % self.config.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"Step {self.global_step}/{total_steps} | "
                        f"Loss: {loss:.4f} | Acc: {accuracy:.2%} | LR: {lr:.2e}"
                    )
                
                if self.global_step % self.config.eval_interval == 0:
                    val_metrics = self.evaluate()
                    if val_metrics:
                        print(f">>> Validation | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.2%}")
                        
                        if val_metrics["accuracy"] > self.best_accuracy:
                            self.best_accuracy = val_metrics["accuracy"]
                            self.save_checkpoint("best_model")
        
        total_time = time.time() - start_time
        print(f"\nReward model training finished! Time: {total_time / 60:.1f} min")
        self.save_checkpoint("final_model")


# =============================================================================
# PPO Trainer
# =============================================================================

@dataclass
class PPOConfig:
    """PPO training config"""
    # Model paths
    sft_model_path: str = ""
    reward_model_path: str = ""
    model_size: str = "small"
    
    # PPO hyperparameters
    ppo_epochs: int = 4  # Number of epochs per batch
    clip_ratio: float = 0.2  # PPO clipping ratio
    value_clip: float = 0.2  # Value function clipping
    kl_coef: float = 0.1  # KL penalty coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    gamma: float = 1.0  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    
    # Generation parameters
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 50
    
    # Training parameters
    batch_size: int = 4
    mini_batch_size: int = 2
    max_seq_len: int = 512
    num_episodes: int = 1000
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100
    output_dir: str = "ppo_run"
    
    # Device
    device: str = "auto"


class ValueHead(nn.Module):
    """Value head: estimates state value"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = F.tanh(self.dense(hidden_states))
        return self.out(x).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Actor-Critic model
    
    Actor: policy model, generates tokens
    Critic: value model, estimates state value
    """
    
    def __init__(self, policy_model: GPT):
        super().__init__()
        self.policy = policy_model
        self.value_head = ValueHead(policy_model.config.d_model)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: [batch, seq_len, vocab_size]
            values: [batch, seq_len]
        """
        # Get hidden states
        x = self.policy.token_embedding(input_ids)
        x = self.policy.dropout(x)
        
        for layer in self.policy.layers:
            x = layer(x, attention_mask)
        
        x = self.policy.final_norm(x)
        
        # Policy logits
        logits = self.policy.lm_head(x)
        
        # State values
        values = self.value_head(x)
        
        return logits, values
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate tokens and return log probs and values
        
        Returns:
            generated_ids: full generated sequence
            log_probs: log prob for each generated token
            values: value estimate for each position
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        generated_ids = input_ids.clone()
        all_log_probs = []
        all_values = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, values = self(generated_ids)
                
                # Only take the last position
                next_logits = logits[:, -1, :] / temperature
                next_value = values[:, -1]
                
                # Top-K sampling
                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float("-inf")
                
                # Sampling
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Compute log prob
                log_probs = F.log_softmax(next_logits, dim=-1)
                next_log_prob = log_probs.gather(dim=-1, index=next_token).squeeze(-1)
                
                # Save
                all_log_probs.append(next_log_prob)
                all_values.append(next_value)
                
                # Concatenate
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Check EOS (simplified)
                if (next_token == 3).all():  # EOS token id = 3
                    break
        
        # Stack
        log_probs = torch.stack(all_log_probs, dim=1) if all_log_probs else torch.zeros(batch_size, 0, device=device)
        values = torch.stack(all_values, dim=1) if all_values else torch.zeros(batch_size, 0, device=device)
        
        return generated_ids, log_probs, values


class PPOTrainer:
    """
    PPO Trainer
    
    Optimizes policy using Proximal Policy Optimization algorithm
    """
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.setup_device()
        self.setup_output_dir()
        self.setup_models()
        self.setup_prompts()
        self.setup_optimizer()
        
        self.global_step = 0
        self.train_logs = []
    
    def setup_device(self):
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
        self.output_dir = OUTPUT_DIR / self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_dir / "ppo_config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        print(f"Output directory: {self.output_dir}")
    
    def setup_models(self):
        self.tokenizer = LLMTokenizer(TOKENIZER_DIR)
        
        # Load policy model
        if self.config.sft_model_path:
            sft_dir = OUTPUT_DIR / self.config.sft_model_path
            print(f"Loading SFT model: {sft_dir}")
            
            with open(sft_dir / "model_config.json", "r") as f:
                model_config_dict = json.load(f)
            model_config = GPTConfig(**model_config_dict)
            
            policy = GPT(model_config)
            checkpoint = torch.load(sft_dir / "best_model.pt", map_location=self.device)
            policy.load_state_dict(checkpoint["model_state_dict"])
            
            # Reference model (frozen)
            ref_policy = GPT(model_config)
            ref_policy.load_state_dict(checkpoint["model_state_dict"])
        else:
            config_map = {
                "tiny": GPTConfig.tiny,
                "small": GPTConfig.small,
                "medium": GPTConfig.medium,
                "large": GPTConfig.large,
            }
            model_config = config_map[self.config.model_size](self.tokenizer.vocab_size)
            model_config.max_seq_len = self.config.max_seq_len
            
            policy = GPT(model_config)
            ref_policy = GPT(model_config)
            ref_policy.load_state_dict(policy.state_dict())
        
        # Actor-Critic
        self.actor_critic = ActorCritic(policy).to(self.device)
        
        # Reference model
        self.ref_policy = ref_policy.to(self.device)
        self.ref_policy.eval()
        for param in self.ref_policy.parameters():
            param.requires_grad = False
        
        # Load reward model
        if self.config.reward_model_path:
            rm_dir = OUTPUT_DIR / self.config.reward_model_path
            print(f"Loading reward model: {rm_dir}")
            
            with open(rm_dir / "model_config.json", "r") as f:
                rm_config_dict = json.load(f)
            rm_config = GPTConfig(**rm_config_dict)
            
            self.reward_model = RewardModel(rm_config)
            rm_checkpoint = torch.load(rm_dir / "best_model.pt", map_location=self.device)
            self.reward_model.load_state_dict(rm_checkpoint["model_state_dict"])
            self.reward_model.to(self.device)
            self.reward_model.eval()
            for param in self.reward_model.parameters():
                param.requires_grad = False
            print("Reward model loaded successfully!")
        else:
            print("Warning: No reward model specified, using random rewards")
            self.reward_model = None
        
        print(f"Policy model parameter count: {policy.count_parameters():,}")
    
    def setup_prompts(self):
        """Load prompts for training"""
        # Extract prompts from preference data
        train_path = DATA_DIR / "preference" / "train.jsonl"
        
        self.prompts = []
        if train_path.exists():
            with open(train_path, "r") as f:
                for line in f:
                    item = json.loads(line.strip())
                    self.prompts.append(item["prompt"])
        
        if not self.prompts:
            # Default prompts
            self.prompts = [
                "What is the capital of France?",
                "Explain machine learning in simple terms.",
                "Write a short poem about nature.",
                "What are the benefits of exercise?",
                "How does the internet work?",
            ] * 100
        
        print(f"Loaded {len(self.prompts)} prompts")
    
    def setup_optimizer(self):
        self.optimizer = AdamW(
            self.actor_critic.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
    
    @torch.no_grad()
    def compute_rewards(
        self,
        generated_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute rewards"""
        if self.reward_model is not None:
            rewards = self.reward_model(generated_ids, attention_mask)
        else:
            # Random rewards (for testing)
            rewards = torch.randn(generated_ids.size(0), device=self.device)
        return rewards
    
    @torch.no_grad()
    def compute_kl_penalty(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence penalty"""
        kl = log_probs - ref_log_probs
        return kl.mean(dim=-1)
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using GAE
        """
        batch_size, seq_len = values.shape
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)
        
        last_gae = 0
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]
            
            delta = rewards[:, t] + self.config.gamma * next_value - values[:, t]
            advantages[:, t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * last_gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def ppo_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> dict:
        """PPO update step"""
        self.actor_critic.train()
        
        # Forward pass
        logits, values = self.actor_critic(input_ids, attention_mask)
        
        # Compute new log probs
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Only take response part
        # Simplified: assume response starts at a certain position in input_ids
        response_logits = logits[:, :-1, :]  # shift
        response_targets = input_ids[:, 1:]
        
        new_log_probs = log_probs[:, :-1, :].gather(
            dim=-1,
            index=response_targets.unsqueeze(-1)
        ).squeeze(-1)
        
        # Clip to actual response length
        actual_len = min(new_log_probs.size(1), old_log_probs.size(1))
        new_log_probs = new_log_probs[:, :actual_len]
        old_log_probs = old_log_probs[:, :actual_len]
        advantages = advantages[:, :actual_len]
        returns = returns[:, :actual_len]
        old_values = old_values[:, :actual_len]
        values = values[:, 1:actual_len + 1]
        
        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value function loss
        value_pred_clipped = old_values + torch.clamp(
            values - old_values,
            -self.config.value_clip,
            self.config.value_clip,
        )
        value_loss1 = (values - returns) ** 2
        value_loss2 = (value_pred_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        
        # Entropy bonus
        entropy = -(F.softmax(response_logits, dim=-1) * F.log_softmax(response_logits, dim=-1)).sum(dim=-1).mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss - self.config.entropy_coef * entropy
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.grad_clip)
        
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "ratio": ratio.mean().item(),
        }
    
    def rollout(self, prompts: List[str]) -> dict:
        """
        Perform one rollout: generate responses and collect data
        """
        self.actor_critic.eval()
        
        # Encode prompts
        batch_prompts = [f"User: {p}\nAssistant: " for p in prompts]
        
        max_prompt_len = 0
        prompt_ids_list = []
        for p in batch_prompts:
            ids = self.tokenizer.encode(p, add_special_tokens=False)
            prompt_ids_list.append(ids)
            max_prompt_len = max(max_prompt_len, len(ids))
        
        # Padding
        padded_prompts = []
        prompt_masks = []
        for ids in prompt_ids_list:
            pad_len = max_prompt_len - len(ids)
            padded = [self.tokenizer.pad_token_id] * pad_len + ids
            mask = [0] * pad_len + [1] * len(ids)
            padded_prompts.append(padded)
            prompt_masks.append(mask)
        
        input_ids = torch.tensor(padded_prompts, device=self.device)
        prompt_mask = torch.tensor(prompt_masks, device=self.device)
        
        # Generate responses
        generated_ids, log_probs, values = self.actor_critic.generate(
            input_ids,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
        )
        
        # Compute attention mask
        attention_mask = torch.ones_like(generated_ids)
        attention_mask[:, :input_ids.size(1)] = prompt_mask
        
        # Compute rewards
        rewards = self.compute_rewards(generated_ids, attention_mask)
        
        # Compute reference model log probs (for KL penalty)
        with torch.no_grad():
            ref_outputs = self.ref_policy(generated_ids, attention_mask)
            ref_logits = ref_outputs["logits"]
            ref_log_probs = F.log_softmax(ref_logits[:, input_ids.size(1)-1:-1, :], dim=-1)
            
            response_tokens = generated_ids[:, input_ids.size(1):]
            actual_len = min(ref_log_probs.size(1), response_tokens.size(1))
            ref_log_probs = ref_log_probs[:, :actual_len, :].gather(
                dim=-1,
                index=response_tokens[:, :actual_len].unsqueeze(-1)
            ).squeeze(-1)
        
        # KL penalty
        actual_len = min(log_probs.size(1), ref_log_probs.size(1))
        kl_penalty = self.compute_kl_penalty(
            log_probs[:, :actual_len],
            ref_log_probs[:, :actual_len]
        )
        
        # Adjusted rewards (reward - KL penalty)
        adjusted_rewards = rewards - self.config.kl_coef * kl_penalty
        
        # Assign rewards to each token (simplified: only last token gets reward)
        token_rewards = torch.zeros_like(values)
        if token_rewards.size(1) > 0:
            token_rewards[:, -1] = adjusted_rewards
        
        # Compute advantages
        dones = torch.zeros_like(values)
        if dones.size(1) > 0:
            dones[:, -1] = 1
        
        advantages, returns = self.compute_advantages(token_rewards, values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            "input_ids": generated_ids,
            "attention_mask": attention_mask,
            "log_probs": log_probs[:, :actual_len],
            "values": values[:, :actual_len],
            "advantages": advantages[:, :actual_len],
            "returns": returns[:, :actual_len],
            "rewards": rewards,
            "kl_penalty": kl_penalty,
            "response_mask": torch.ones(values.size(0), actual_len, device=self.device),
        }
    
    def save_checkpoint(self, name: str):
        checkpoint = {
            "actor_critic_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }
        torch.save(checkpoint, self.output_dir / f"{name}.pt")
        
        # Also save pure policy model (for inference)
        policy_checkpoint = {
            "model_state_dict": self.actor_critic.policy.state_dict(),
        }
        torch.save(policy_checkpoint, self.output_dir / f"{name}_policy.pt")
        
        with open(self.output_dir / "model_config.json", "w") as f:
            json.dump(asdict(self.actor_critic.policy.config), f, indent=2)
        
        print(f"Checkpoint saved: {self.output_dir / name}.pt")
    
    @torch.no_grad()
    def generate_sample(self, prompt: str) -> str:
        """Generate sample"""
        self.actor_critic.eval()
        
        full_prompt = f"User: {prompt}\nAssistant: "
        input_ids = torch.tensor(
            [self.tokenizer.encode(full_prompt, add_special_tokens=False)],
            device=self.device
        )
        
        generated, _, _ = self.actor_critic.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.7,
            top_k=50,
        )
        
        response = self.tokenizer.decode(generated[0].tolist())
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response
    
    def train(self):
        print("\n" + "=" * 60)
        print("Start PPO training")
        print(f"KL coefficient: {self.config.kl_coef}")
        print(f"Clip ratio: {self.config.clip_ratio}")
        print("=" * 60)
        
        start_time = time.time()
        
        for episode in range(self.config.num_episodes):
            # Sample prompts
            batch_indices = torch.randint(0, len(self.prompts), (self.config.batch_size,))
            batch_prompts = [self.prompts[i] for i in batch_indices]
            
            # Rollout
            rollout_data = self.rollout(batch_prompts)
            
            # PPO update
            total_metrics = {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
            }
            
            for _ in range(self.config.ppo_epochs):
                metrics = self.ppo_step(
                    rollout_data["input_ids"],
                    rollout_data["attention_mask"],
                    rollout_data["log_probs"],
                    rollout_data["values"],
                    rollout_data["advantages"],
                    rollout_data["returns"],
                    rollout_data["response_mask"],
                )
                
                for k, v in metrics.items():
                    if k in total_metrics:
                        total_metrics[k] += v
            
            # Average
            for k in total_metrics:
                total_metrics[k] /= self.config.ppo_epochs
            
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                avg_reward = rollout_data["rewards"].mean().item()
                avg_kl = rollout_data["kl_penalty"].mean().item()
                elapsed = time.time() - start_time
                
                print(
                    f"Episode {self.global_step}/{self.config.num_episodes} | "
                    f"Reward: {avg_reward:.4f} | "
                    f"KL: {avg_kl:.4f} | "
                    f"Loss: {total_metrics['loss']:.4f} | "
                    f"Time: {elapsed:.0f}s"
                )
                
                self.train_logs.append({
                    "step": self.global_step,
                    "reward": avg_reward,
                    "kl": avg_kl,
                    **total_metrics,
                })
                
                # Generate sample
                sample = self.generate_sample("What is machine learning?")
                print(f">>> Sample: {sample[:150]}...")
            
            # Save
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}")
        
        total_time = time.time() - start_time
        print(f"\nPPO training finished! Time: {total_time / 60:.1f} min")
        
        self.save_checkpoint("final_model")
        
        with open(self.output_dir / "training_logs.json", "w") as f:
            json.dump(self.train_logs, f, indent=2)


# =============================================================================
# Main function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RLHF Training")
    
    subparsers = parser.add_subparsers(dest="stage", help="Training stage")
    
    # Reward model training
    rm_parser = subparsers.add_parser("reward", help="Train reward model")
    rm_parser.add_argument("--sft_model_path", type=str, default="")
    rm_parser.add_argument("--model_size", type=str, default="small")
    rm_parser.add_argument("--batch_size", type=int, default=4)
    rm_parser.add_argument("--num_epochs", type=int, default=1)
    rm_parser.add_argument("--learning_rate", type=float, default=1e-5)
    rm_parser.add_argument("--output_dir", type=str, default="reward_model_run")
    rm_parser.add_argument("--device", type=str, default="auto")
    
    # PPO training
    ppo_parser = subparsers.add_parser("ppo", help="PPO training")
    ppo_parser.add_argument("--sft_model_path", type=str, default="")
    ppo_parser.add_argument("--reward_model_path", type=str, default="")
    ppo_parser.add_argument("--model_size", type=str, default="small")
    ppo_parser.add_argument("--batch_size", type=int, default=4)
    ppo_parser.add_argument("--num_episodes", type=int, default=1000)
    ppo_parser.add_argument("--learning_rate", type=float, default=1e-6)
    ppo_parser.add_argument("--kl_coef", type=float, default=0.1)
    ppo_parser.add_argument("--clip_ratio", type=float, default=0.2)
    ppo_parser.add_argument("--output_dir", type=str, default="ppo_run")
    ppo_parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    if args.stage == "reward":
        config = RewardModelConfig(
            sft_model_path=args.sft_model_path,
            model_size=args.model_size,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            device=args.device,
        )
        trainer = RewardModelTrainer(config)
        trainer.train()
    
    elif args.stage == "ppo":
        config = PPOConfig(
            sft_model_path=args.sft_model_path,
            reward_model_path=args.reward_model_path,
            model_size=args.model_size,
            batch_size=args.batch_size,
            num_episodes=args.num_episodes,
            learning_rate=args.learning_rate,
            kl_coef=args.kl_coef,
            clip_ratio=args.clip_ratio,
            output_dir=args.output_dir,
            device=args.device,
        )
        trainer = PPOTrainer(config)
        trainer.train()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()