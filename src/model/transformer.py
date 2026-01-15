"""
GPT Model Definition
Based on Decoder-Only Transformer architecture, supports multiple scale configurations
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    """GPT model configuration"""
    vocab_size: int = 8000          # Vocabulary size
    max_seq_len: int = 512          # Maximum sequence length
    d_model: int = 512              # Model dimension
    n_layers: int = 8               # Number of Transformer layers
    n_heads: int = 8                # Number of attention heads
    d_ff: int = 2048                # FFN hidden dimension
    dropout: float = 0.1            # Dropout rate
    bias: bool = False              # Whether to use bias (modern LLMs usually do not)
    
    # Special token IDs
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3
    
    @classmethod
    def tiny(cls, vocab_size: int = 8000):
        """~10M parameters, for quick debugging"""
        return cls(
            vocab_size=vocab_size,
            d_model=256,
            n_layers=4,
            n_heads=4,
            d_ff=1024,
        )
    
    @classmethod
    def small(cls, vocab_size: int = 8000):
        """~50M parameters, suitable for single GPU training"""
        return cls(
            vocab_size=vocab_size,
            d_model=512,
            n_layers=8,
            n_heads=8,
            d_ff=2048,
        )
    
    @classmethod
    def medium(cls, vocab_size: int = 8000):
        """~150M parameters"""
        return cls(
            vocab_size=vocab_size,
            d_model=768,
            n_layers=12,
            n_heads=12,
            d_ff=3072,
        )
    
    @classmethod
    def large(cls, vocab_size: int = 8000):
        """~350M parameters"""
        return cls(
            vocab_size=vocab_size,
            d_model=1024,
            n_layers=24,
            n_heads=16,
            d_ff=4096,
        )


class RMSNorm(nn.Module):
    """
    RMSNorm - normalization method commonly used in modern LLMs
    Faster than LayerNorm, similar effectiveness
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE - Rotary Positional Embedding
    Standard positional encoding for modern LLMs (LLaMA, Qwen, etc.)
    """
    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Compute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Precompute positional encoding cache"""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]
        
        # Expand to [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin cache"""
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embedding to Q and K
    """
    def rotate_half(x):
        """Negate the second half and swap with the first half"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    # [batch, n_heads, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    Supports RoPE and causal mask
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        
        # QKV projection (merged into one matrix for efficiency)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        
        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(self.head_dim, config.max_seq_len)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len] or None
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Compute Q, K, V
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3 * d_model]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, n_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 2. Apply RoPE
        cos, sin = self.rope(x, seq_len)
        # Expand dimensions to match [batch, n_heads, seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_emb(q, k, cos, sin)
        
        # 3. Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # 4. Apply causal mask (lower triangular matrix)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        
        # 5. Apply padding mask
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(padding_mask, float("-inf"))
        
        # 6. Softmax + Dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 7. Weighted sum
        output = torch.matmul(attn_weights, v)  # [batch, n_heads, seq_len, head_dim]
        
        # 8. Merge heads
        output = output.transpose(1, 2).contiguous()  # [batch, seq_len, n_heads, head_dim]
        output = output.reshape(batch_size, seq_len, -1)  # [batch, seq_len, d_model]
        
        # 9. Output projection
        output = self.out_proj(output)
        output = self.resid_dropout(output)
        
        return output


class FeedForward(nn.Module):
    """
    Feed-forward neural network
    Uses SwiGLU activation (standard for modern LLMs)
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        # SwiGLU requires 3 linear layers
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (x * W1 * SiLU) * (x * W3)
        swish = F.silu(self.w1(x))
        gate = self.w3(x)
        output = swish * gate
        output = self.w2(output)
        output = self.dropout(output)
        return output


class TransformerBlock(nn.Module):
    """
    Transformer decoder layer
    Pre-Norm architecture (standard for modern LLMs)
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-Norm + residual connection
        x = x + self.attn(self.attn_norm(x), attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GPT(nn.Module):
    """
    GPT Model (Decoder-Only Transformer)
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token Embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final normalization
        self.final_norm = RMSNorm(config.d_model)
        
        # Output layer (shared weights with embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight sharing
        self.token_embedding.weight = self.lm_head.weight
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] for loss calculation
        Returns:
            dict with 'logits' and optional 'loss'
        """
        # 1. Token Embedding
        x = self.token_embedding(input_ids)  # [batch, seq_len, d_model]
        x = self.dropout(x)
        
        # 2. Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # 3. Final normalization
        x = self.final_norm(x)
        
        # 4. Compute logits
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        # 5. Compute loss (if labels are provided)
        loss = None
        if labels is not None:
            # Shift logits and labels left by one (predict next token)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id
            )
        
        return {"logits": logits, "loss": loss}
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation
        
        Args:
            input_ids: [batch_size, seq_len] input token IDs
            max_new_tokens: maximum number of generated tokens
            temperature: temperature parameter, higher is more random
            top_k: sample only from top k tokens by probability
            top_p: nucleus sampling, cumulative probability threshold
            eos_token_id: end token ID
        """
        self.eval()
        eos_token_id = eos_token_id or self.config.eos_token_id
        
        for _ in range(max_new_tokens):
            # Truncate to maximum sequence length
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            
            # Forward pass
            outputs = self(idx_cond)
            logits = outputs["logits"][:, -1, :]  # Only last position
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-K sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            
            # Top-P (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")
            
            # Sampling
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Concatenate
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for end
            if (next_token == eos_token_id).all():
                break
        
        return input_ids
    
    def count_parameters(self) -> int:
        """Count model parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_model_info(self):
        """Print model information"""
        total_params = self.count_parameters()
        print("=" * 60)
        print("GPT Model Info")
        print("=" * 60)
        print(f"Vocabulary size: {self.config.vocab_size}")
        print(f"Max sequence length: {self.config.max_seq_len}")
        print(f"Model dimension (d_model): {self.config.d_model}")
        print(f"Number of layers (n_layers): {self.config.n_layers}")
        print(f"Number of attention heads (n_heads): {self.config.n_heads}")
        print(f"FFN dimension (d_ff): {self.config.d_ff}")
        print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
        print("=" * 60)


def demo():
    """Model demo"""
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.data.tokenizer import LLMTokenizer, TOKENIZER_DIR
    
    # 1. Load Tokenizer
    tokenizer = LLMTokenizer(TOKENIZER_DIR)
    
    # 2. Create models of different scales
    configs = [
        ("Tiny (~10M)", GPTConfig.tiny(tokenizer.vocab_size)),
        ("Small (~50M)", GPTConfig.small(tokenizer.vocab_size)),
        ("Medium (~150M)", GPTConfig.medium(tokenizer.vocab_size)),
    ]
    
    for name, config in configs:
        print(f"\n{name} model:")
        model = GPT(config)
        model.print_model_info()
    
    # 3. Test forward pass
    print("\n" + "=" * 60)
    print("Forward pass test")
    print("=" * 60)
    
    config = GPTConfig.small(tokenizer.vocab_size)
    model = GPT(config)
    
    # Prepare input
    texts = ["Hello, world!", "The quick brown fox"]
    batch = tokenizer.batch_encode(texts, max_length=32)
    
    print(f"Input shape: {batch['input_ids'].shape}")
    
    # Forward pass
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['input_ids']  # Autoregressive training, labels = input_ids
    )
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    # 4. Test generation (untrained model will generate random content)
    print("\n" + "=" * 60)
    print("Generation test (untrained, output is random)")
    print("=" * 60)
    
    prompt = "The"
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)])
    
    generated = model.generate(
        input_ids,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50
    )
    
    print(f"Input: {prompt}")
    print(f"Generated: {tokenizer.decode(generated[0].tolist())}")


if __name__ == "__main__":
    from pathlib import Path
    demo()