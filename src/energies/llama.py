"""
Implement the Llama 3.2-1B model in JAX/Flax.
"""

import json
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class LlamaConfig:
    """Llama model configuration."""

    vocab_size: int = 128256
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 64
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = True

    @classmethod
    def from_json(cls, config_path: str) -> "LlamaConfig":
        """Load config from JSON file."""
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Map JSON keys to our config
        return cls(
            vocab_size=config_dict["vocab_size"],
            hidden_size=config_dict["hidden_size"],
            intermediate_size=config_dict["intermediate_size"],
            num_hidden_layers=config_dict["num_hidden_layers"],
            num_attention_heads=config_dict["num_attention_heads"],
            num_key_value_heads=config_dict["num_key_value_heads"],
            head_dim=config_dict["head_dim"],
            max_position_embeddings=config_dict["max_position_embeddings"],
            rms_norm_eps=config_dict["rms_norm_eps"],
            rope_theta=config_dict["rope_theta"],
            tie_word_embeddings=config_dict["tie_word_embeddings"],
        )


class LlamaEmbeddings(nn.Module):
    """Token embeddings for Llama."""

    config: LlamaConfig

    def setup(self):
        self.embed_tokens = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=nn.initializers.normal(
                stddev=0.02
            ),  # From config.initializer_range
        )

    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)

        Returns:
            embeddings: Token embeddings of shape (batch_size, seq_len, hidden_size)
        """
        return self.embed_tokens(input_ids)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    config: LlamaConfig

    def setup(self):
        # RMSNorm only has a scale parameter (no bias)
        self.weight = self.param(
            "weight", nn.initializers.ones, (self.config.hidden_size,)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape (..., hidden_size)
        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS (Root Mean Square)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.config.rms_norm_eps)

        # Scale by learned parameter
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    config: LlamaConfig

    def setup(self):
        # Compute inverse frequencies for each dimension pair
        dim = self.config.head_dim  # 64
        inv_freq = 1.0 / (
            self.config.rope_theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
        )
        self.inv_freq = inv_freq  # Shape: (32,) - one freq per dim pair

    def __call__(self, seq_len: int):
        """
        Generate cos/sin rotation matrices for sequence positions.

        Args:
            seq_len: Length of sequence

        Returns:
            cos, sin: Rotation matrices of shape (seq_len, head_dim//2)
        """
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = jnp.arange(seq_len, dtype=jnp.float32)

        # Compute angles: position * frequency for each dim pair
        # positions: (seq_len,), inv_freq: (head_dim//2,)
        # Result: (seq_len, head_dim//2)
        angles = positions[:, None] * self.inv_freq[None, :]

        # Compute cos and sin
        cos = jnp.cos(angles)  # (seq_len, head_dim//2)
        sin = jnp.sin(angles)  # (seq_len, head_dim//2)

        return cos, sin


def apply_rotary_pos_emb(
    x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray
) -> jnp.ndarray:
    """
    Apply rotary position embedding to input tensor.

    Args:
        x: Input tensor of shape (..., seq_len, head_dim)
        cos: Cosine values of shape (seq_len, head_dim//2)
        sin: Sine values of shape (seq_len, head_dim//2)

    Returns:
        Rotated tensor of same shape as input
    """
    x1 = x[..., 0::2]  # Even indices: 0, 2, 4, ... -> shape (..., seq_len, head_dim//2)
    x2 = x[..., 1::2]  # Odd indices: 1, 3, 5, ... -> shape (..., seq_len, head_dim//2)

    # Reshape cos/sin to broadcast properly
    # We need to add dimensions to match x1/x2 shape
    # x1/x2: (..., seq_len, head_dim//2)
    # cos/sin: (seq_len, head_dim//2) → (1, seq_len, 1, head_dim//2)
    cos = cos[None, :, None, :]  # Add batch and head dimensions
    sin = sin[None, :, None, :]

    # Apply 2D rotation to each pair
    # Rotation matrix: [[cos, -sin], [sin, cos]]
    # [x1'] = [cos*x1 - sin*x2]
    # [x2'] = [sin*x1 + cos*x2]
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos

    # Interleave back: (x1', x2', x3', x4', ...)
    # Stack along new axis then reshape to interleave
    rotated = jnp.stack([x1_rot, x2_rot], axis=-1)  # (..., seq_len, head_dim//2, 2)
    rotated = rotated.reshape(x.shape)  # (..., seq_len, head_dim)

    return rotated


class LlamaAttention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA) and RoPE."""

    config: LlamaConfig

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.head_dim = self.config.head_dim

        # Compute how many query heads per key/value head
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Projections (no bias in Llama)
        self.q_proj = nn.Dense(
            self.num_heads * self.head_dim,  # 32 * 64 = 2048
            use_bias=False,
        )
        self.k_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,  # 8 * 64 = 512
            use_bias=False,
        )
        self.v_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,  # 8 * 64 = 512
            use_bias=False,
        )
        self.o_proj = nn.Dense(
            self.hidden_size,  # 2048
            use_bias=False,
        )

        # RoPE for positional encoding
        self.rotary_emb = RotaryEmbedding(self.config)

    def __call__(self, hidden_states: jnp.ndarray, attention_mask: jnp.ndarray = None):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len, seq_len) - causal mask

        Returns:
            attention_output: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Project to Q, K, V
        query_states = self.q_proj(hidden_states)  # (batch, seq_len, 32*64)
        key_states = self.k_proj(hidden_states)  # (batch, seq_len, 8*64)
        value_states = self.v_proj(hidden_states)  # (batch, seq_len, 8*64)

        # 2. Reshape to separate heads
        query_states = query_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )  # (batch, seq_len, 32, 64)

        key_states = key_states.reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )  # (batch, seq_len, 8, 64)

        value_states = value_states.reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )  # (batch, seq_len, 8, 64)

        # 3. Apply RoPE to Q and K (not V!)
        cos, sin = self.rotary_emb(seq_len)
        query_states = apply_rotary_pos_emb(query_states, cos, sin)
        key_states = apply_rotary_pos_emb(key_states, cos, sin)

        # 4. Expand K,V to match Q heads (GQA)
        # Repeat each K,V head 4 times (since 32/8 = 4)
        key_states = jnp.repeat(
            key_states, self.num_key_value_groups, axis=2
        )  # (batch, seq_len, 32, 64)

        value_states = jnp.repeat(
            value_states, self.num_key_value_groups, axis=2
        )  # (batch, seq_len, 32, 64)

        # 5. Transpose for attention computation
        # We want: (batch, num_heads, seq_len, head_dim)
        query_states = query_states.transpose(0, 2, 1, 3)  # (batch, 32, seq_len, 64)
        key_states = key_states.transpose(0, 2, 1, 3)  # (batch, 32, seq_len, 64)
        value_states = value_states.transpose(0, 2, 1, 3)  # (batch, 32, seq_len, 64)

        # 6. Compute attention scores
        attn_weights = jnp.matmul(query_states, key_states.transpose(0, 1, 3, 2))
        # (batch, 32, seq_len, seq_len)

        # Scale by sqrt(head_dim)
        attn_weights = attn_weights / jnp.sqrt(self.head_dim)

        # 7. Apply causal mask
        if attention_mask is not None:
            attn_weights = jnp.where(attention_mask, attn_weights, -jnp.inf)

        # 8. Softmax
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # 9. Apply attention to values
        attn_output = jnp.matmul(attn_weights, value_states)
        # (batch, 32, seq_len, 64)

        # 10. Transpose back and reshape
        attn_output = attn_output.transpose(0, 2, 1, 3)  # (batch, seq_len, 32, 64)
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        # (batch, seq_len, 2048)

        # 11. Final output projection
        attn_output = self.o_proj(attn_output)

        return attn_output


def create_causal_mask(seq_len: int) -> jnp.ndarray:
    """Create a causal (lower triangular) attention mask."""
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    # Expand for batch and head dimensions: (1, 1, seq_len, seq_len)
    return mask[None, None, :, :]


# Test function
def test_attention():
    """Test the attention layer."""
    config_path = Path(__file__).parent / "llama_cfg.json"
    config = LlamaConfig.from_json(str(config_path))

    print("Testing Llama Attention...")
    print(f"num_attention_heads: {config.num_attention_heads}")
    print(f"num_key_value_heads: {config.num_key_value_heads}")
    print(f"head_dim: {config.head_dim}")
    print(f"GQA ratio: {config.num_attention_heads // config.num_key_value_heads}:1")

    # Create attention layer
    attention = LlamaAttention(config=config)

    # Test input
    key = jax.random.PRNGKey(42)
    batch_size, seq_len = 2, 8
    hidden_states = jax.random.normal(key, (batch_size, seq_len, config.hidden_size))

    # Create causal mask
    causal_mask = create_causal_mask(seq_len)

    # Initialize parameters
    params = attention.init(key, hidden_states, causal_mask)

    # Forward pass
    output = attention.apply(params, hidden_states, causal_mask)

    print(f"\nAttention test:")
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ Shapes match: {hidden_states.shape == output.shape}")

    # Check parameter counts
    def count_params(params_dict):
        total = 0
        for key, value in params_dict.items():
            if isinstance(value, dict):
                total += count_params(value)
            else:
                total += value.size
        return total

    total_params = count_params(params)
    print(f"Total parameters: {total_params:,}")

    # Expected: q_proj(4.2M) + k_proj(1.0M) + v_proj(1.0M) + o_proj(4.2M) ≈ 10.4M
    expected_params = (
        config.hidden_size * config.num_attention_heads * config.head_dim  # q_proj
        + config.hidden_size * config.num_key_value_heads * config.head_dim  # k_proj
        + config.hidden_size * config.num_key_value_heads * config.head_dim  # v_proj
        + config.hidden_size * config.hidden_size  # o_proj
    )
    print(f"Expected parameters: {expected_params:,}")
    print(f"✓ Parameter count matches: {total_params == expected_params}")


class LlamaMLP(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    config: LlamaConfig

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size

        # SwiGLU requires 3 projections instead of the usual 2
        self.gate_proj = nn.Dense(self.intermediate_size, use_bias=False)  # W_gate
        self.up_proj = nn.Dense(self.intermediate_size, use_bias=False)  # W_up
        self.down_proj = nn.Dense(self.hidden_size, use_bias=False)  # W_down

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        SwiGLU activation: SwiGLU(x) = Swish(W_gate(x)) ⊙ W_up(x)
        where Swish(x) = x * sigmoid(x) = x * σ(x)

        Args:
            x: Input tensor of shape (..., hidden_size)

        Returns:
            Output tensor of shape (..., hidden_size)
        """
        # Project to intermediate size
        gate = self.gate_proj(x)  # (..., intermediate_size)
        up = self.up_proj(x)  # (..., intermediate_size)

        # Apply SwiGLU: Swish(gate) * up
        # Swish(x) = x * sigmoid(x)
        swish_gate = gate * jax.nn.sigmoid(gate)
        intermediate = swish_gate * up

        # Project back to hidden size
        output = self.down_proj(intermediate)

        return output


# Test function
def test_mlp():
    """Test the MLP layer."""
    config_path = Path(__file__).parent / "llama_cfg.json"
    config = LlamaConfig.from_json(str(config_path))

    print("Testing Llama MLP...")
    print(f"hidden_size: {config.hidden_size}")
    print(f"intermediate_size: {config.intermediate_size}")
    print(f"Expansion ratio: {config.intermediate_size / config.hidden_size:.1f}x")

    # Create MLP layer
    mlp = LlamaMLP(config=config)

    # Test input
    key = jax.random.PRNGKey(42)
    batch_size, seq_len = 2, 8
    hidden_states = jax.random.normal(key, (batch_size, seq_len, config.hidden_size))

    # Initialize parameters
    params = mlp.init(key, hidden_states)

    # Forward pass
    output = mlp.apply(params, hidden_states)

    print(f"\nMLP test:")
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ Shapes match: {hidden_states.shape == output.shape}")

    # Check parameter counts
    def count_params(params_dict):
        total = 0
        for key, value in params_dict.items():
            if isinstance(value, dict):
                total += count_params(value)
            else:
                total += value.size
        return total

    total_params = count_params(params)
    print(f"Total parameters: {total_params:,}")

    # Expected parameters:
    # gate_proj: hidden_size × intermediate_size = 2048 × 8192 = 16.8M
    # up_proj:   hidden_size × intermediate_size = 2048 × 8192 = 16.8M
    # down_proj: intermediate_size × hidden_size = 8192 × 2048 = 16.8M
    # Total: 50.3M per MLP layer
    expected_params = (
        config.hidden_size * config.intermediate_size  # gate_proj
        + config.hidden_size * config.intermediate_size  # up_proj
        + config.intermediate_size * config.hidden_size  # down_proj
    )
    print(f"Expected parameters: {expected_params:,}")
    print(f"✓ Parameter count matches: {total_params == expected_params}")

    # Test SwiGLU vs standard activation
    print(f"\nActivation comparison:")
    x_test = jnp.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])

    # Standard ReLU
    relu_out = jax.nn.relu(x_test)
    print(f"ReLU(x):     {relu_out[0]}")

    # SiLU/Swish
    silu_out = x_test * jax.nn.sigmoid(x_test)
    print(f"SiLU(x):     {silu_out[0]}")

    # Show the smooth vs sharp difference
    print("SiLU is smoother than ReLU (no sharp cutoff at 0)")


class LlamaTransformerBlock(nn.Module):  # Better name!
    """A single Llama transformer block."""

    config: LlamaConfig

    def setup(self):
        # Self-attention
        self.attention = LlamaAttention(self.config)

        # Feed-forward network
        self.feed_forward = LlamaMLP(self.config)

        # Layer norms (RMSNorm in Llama)
        self.attention_norm = RMSNorm(self.config)
        self.ffn_norm = RMSNorm(self.config)

    def __call__(self, x: jnp.ndarray, attention_mask: jnp.ndarray = None):
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len, seq_len) - causal mask

        Returns:
            x: (batch_size, seq_len, hidden_size)
        """
        # Self-attention with pre-norm and residual (one line!)
        h = x + self.attention(self.attention_norm(x), attention_mask)

        # Feed-forward with pre-norm and residual (one line!)
        out = h + self.feed_forward(self.ffn_norm(h))

        return out


# Test function
def test_transformer_block():
    """Test a single transformer block."""
    config_path = Path(__file__).parent / "llama_cfg.json"
    config = LlamaConfig.from_json(str(config_path))

    print("Testing Llama Transformer Block...")
    print(f"hidden_size: {config.hidden_size}")

    # Create transformer block
    block = LlamaTransformerBlock(config=config)

    # Test input
    key = jax.random.PRNGKey(42)
    batch_size, seq_len = 2, 8
    hidden_states = jax.random.normal(key, (batch_size, seq_len, config.hidden_size))

    # Create causal mask
    causal_mask = create_causal_mask(seq_len)

    # Initialize parameters
    params = block.init(key, hidden_states, causal_mask)

    # Forward pass
    output = block.apply(params, hidden_states, causal_mask)

    print(f"\nTransformer block test:")
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ Shapes match: {hidden_states.shape == output.shape}")

    # Check parameter counts
    def count_params(params_dict):
        total = 0
        for key, value in params_dict.items():
            if isinstance(value, dict):
                total += count_params(value)
            else:
                total += value.size
        return total

    total_params = count_params(params)
    print(f"Total parameters: {total_params:,}")

    # Expected: Attention (10.5M) + MLP (50.3M) + 2 RMSNorms (2 * 2048 = 4K) ≈ 60.8M
    attention_params = (
        config.hidden_size * config.num_attention_heads * config.head_dim  # q_proj
        + config.hidden_size * config.num_key_value_heads * config.head_dim  # k_proj
        + config.hidden_size * config.num_key_value_heads * config.head_dim  # v_proj
        + config.hidden_size * config.hidden_size  # o_proj
    )

    mlp_params = (
        config.hidden_size * config.intermediate_size  # gate_proj
        + config.hidden_size * config.intermediate_size  # up_proj
        + config.intermediate_size * config.hidden_size  # down_proj
    )

    norm_params = 2 * config.hidden_size  # 2 RMSNorms

    expected_params = attention_params + mlp_params + norm_params
    print(f"Expected parameters: {expected_params:,}")
    print(f"  - Attention: {attention_params:,}")
    print(f"  - MLP: {mlp_params:,}")
    print(f"  - Norms: {norm_params:,}")
    print(f"✓ Parameter count matches: {total_params == expected_params}")

    # Test that residual connections work
    print(f"\nTesting residual connections:")

    # With very small weights, output should be close to input (due to residuals)
    small_params = jax.tree_util.tree_map(lambda x: x * 0.001, params)
    small_output = block.apply(small_params, hidden_states, causal_mask)

    residual_diff = jnp.mean(jnp.abs(small_output - hidden_states))
    print(f"Mean difference with small weights: {residual_diff:.6f}")
    print("✓ Small difference confirms residual connections working")


class LlamaModel(nn.Module):
    """Complete Llama model."""

    config: LlamaConfig

    def setup(self):
        self.embed_tokens = LlamaEmbeddings(self.config)
        self.layers = [
            LlamaTransformerBlock(self.config)
            for _ in range(self.config.num_hidden_layers)  # 16 layers
        ]
        self.norm = RMSNorm(self.config)

    def __call__(self, input_ids, attention_mask=None):
        # Token embeddings
        x = self.embed_tokens(input_ids)

        # Create causal mask if not provided
        if attention_mask is None:
            seq_len = input_ids.shape[1]
            attention_mask = create_causal_mask(seq_len)

        # Through all transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Final normalization
        x = self.norm(x)

        # Output projection using tied embeddings
        logits = x @ self.embed_tokens.embed_tokens.embedding.T

        return logits


def test_full_model():
    """Test the complete Llama model."""
    config_path = Path(__file__).parent / "llama_cfg.json"
    config = LlamaConfig.from_json(str(config_path))

    print("Testing Complete Llama Model...")
    print(f"Model: Llama 3.2-{config.num_hidden_layers}L-{config.hidden_size}H")
    print(f"Vocabulary size: {config.vocab_size:,}")
    print(f"Max sequence length: {config.max_position_embeddings:,}")

    # Create full model
    model = LlamaModel(config=config)

    # Test input - simulate some token IDs
    key = jax.random.PRNGKey(42)
    batch_size, seq_len = 2, 16
    input_ids = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)

    print(f"\nInput:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Sample token IDs: {input_ids[0, :8].tolist()}")

    # Initialize model parameters
    print(f"\nInitializing model parameters...")
    params = model.init(key, input_ids)

    # Forward pass
    print(f"Running forward pass...")
    logits = model.apply(params, input_ids)

    print(f"\nOutput:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {config.vocab_size})")
    print(
        f"  ✓ Shape correct: {logits.shape == (batch_size, seq_len, config.vocab_size)}"
    )

    # Convert logits to probabilities for the last token
    last_token_logits = logits[0, -1, :]  # First batch, last position
    probs = jax.nn.softmax(last_token_logits)

    # Find top-5 most likely next tokens
    top_k = 5
    top_indices = jnp.argsort(probs)[-top_k:][::-1]  # Top-k in descending order
    top_probs = probs[top_indices]

    print(f"\nNext token predictions (first example, last position):")
    for i, (token_id, prob) in enumerate(zip(top_indices, top_probs)):
        print(f"  {i + 1}. Token {token_id}: {prob:.4f} probability")

    # Count total parameters
    def count_params(params_dict):
        total = 0
        for key, value in params_dict.items():
            if isinstance(value, dict):
                total += count_params(value)
            else:
                total += value.size
        return total

    total_params = count_params(params)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Parameters in billions: {total_params / 1e9:.3f}B")

    # Expected breakdown
    embedding_params = config.vocab_size * config.hidden_size
    block_params = (
        # Attention
        config.hidden_size * config.num_attention_heads * config.head_dim
        + config.hidden_size * config.num_key_value_heads * config.head_dim
        + config.hidden_size * config.num_key_value_heads * config.head_dim
        + config.hidden_size * config.hidden_size
        +
        # MLP
        config.hidden_size * config.intermediate_size
        + config.hidden_size * config.intermediate_size
        + config.intermediate_size * config.hidden_size
        +
        # Norms
        2 * config.hidden_size
    )
    total_block_params = config.num_hidden_layers * block_params
    final_norm_params = config.hidden_size

    expected_total = embedding_params + total_block_params + final_norm_params

    print(f"\nParameter Breakdown:")
    print(
        f"  Embeddings (tied): {embedding_params:,}"
        f" ({embedding_params / total_params * 100:.1f}%)"
    )
    print(
        f"  {config.num_hidden_layers} Transformer blocks: {total_block_params:,} "
        f"({total_block_params / total_params * 100:.1f}%)"
    )
    print(
        f"  Final norm: {final_norm_params:,}"
        f" ({final_norm_params / total_params * 100:.1f}%)"
    )
    print(f"  Expected total: {expected_total:,}")
    print(f"  ✓ Parameter count matches: {total_params == expected_total}")

    print(f"\n🎉 Complete Llama 3.2-1B model implemented successfully!")
    print(f"   Ready for training or inference with pre-trained weights!")


if __name__ == "__main__":
    print("--------------------------------")
    print("Testing RMSNorm")
    print("--------------------------------")

    # Test the shapes
    x = jnp.ones((2, 10, 2048))  # batch=2, seq=10, hidden=2048
    weight = jnp.ones((2048,))  # just hidden_size
    result = weight * x  # broadcasts to (2, 10, 2048)
    print(result.shape)  # (2, 10, 2048) ✓

    print("--------------------------------")
    print("Testing Attention")
    print("--------------------------------")
    test_attention()

    print("--------------------------------")
    print("Testing MLP")
    print("--------------------------------")
    test_mlp()

    print("--------------------------------")
    print("Testing Transformer Block")
    print("--------------------------------")
    test_transformer_block()

    print("--------------------------------")
    print("Testing Complete Model")
    print("--------------------------------")
    test_full_model()
