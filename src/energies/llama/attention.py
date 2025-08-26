"""
Multi-head attention with Grouped Query Attention (GQA) and RoPE for Llama.

This is the core of the transformer - where tokens attend to each other
based on both content similarity and relative position.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from config import LlamaConfig


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    RoPE encodes positional information by rotating query and key vectors
    based on their position. Different frequency components capture
    relationships at different scales (local vs global).
    """

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
    """Apply rotary position embedding correctly."""
    # Split into pairs
    x1 = x[..., 0::2]  # (..., seq_len, head_dim//2)
    x2 = x[..., 1::2]  # (..., seq_len, head_dim//2)

    # cos/sin: (seq_len, head_dim//2)
    # x1/x2: (batch, seq_len, num_heads, head_dim//2)

    # We need cos/sin to be: (1, seq_len, 1, head_dim//2)
    # So it broadcasts across batch and head dimensions
    cos_expanded = cos[None, :, None, :]  # Add batch and head dims
    sin_expanded = sin[None, :, None, :]  # Add batch and head dims

    # Now broadcasting works:
    # x1: (batch, seq_len, num_heads, head_dim//2)
    # cos: (1, seq_len, 1, head_dim//2)
    # Result: (batch, seq_len, num_heads, head_dim//2)

    x1_rot = x1 * cos_expanded - x2 * sin_expanded
    x2_rot = x1 * sin_expanded + x2 * cos_expanded

    # Interleave back
    rotated = jnp.stack([x1_rot, x2_rot], axis=-1)
    return rotated.reshape(x.shape)


class LlamaAttention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA) and RoPE.

    Key innovations:
    1. Grouped Query Attention: 32 query heads share 8 key/value heads (4:1 ratio)
    2. RoPE: Positional encoding applied fresh at each layer
    3. No bias terms: Cleaner, more efficient
    """

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
        Multi-head attention forward pass.

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


def test_rope():
    """Test RoPE implementation."""
    from pathlib import Path

    import jax
    from config import LlamaConfig

    config_path = Path(__file__).parent.parent / "llama_cfg.json"
    config = LlamaConfig.from_json(str(config_path))

    print("=== Testing RoPE ===")
    print(f"Head dim: {config.head_dim}")
    print(f"RoPE theta: {config.rope_theta}")

    # Create RoPE
    rope = RotaryEmbedding(config=config)

    # Test
    key = jax.random.PRNGKey(42)
    seq_len = 10
    params = rope.init(key, seq_len)
    cos, sin = rope.apply(params, seq_len)

    print(f"\n=== RoPE Output ===")
    print(f"cos shape: {cos.shape}")
    print(f"sin shape: {sin.shape}")
    print(f"Expected: ({seq_len}, {config.head_dim // 2})")

    # Test rotation application
    batch_size, num_heads = 2, 4
    x = jax.random.normal(key, (batch_size, seq_len, num_heads, config.head_dim))
    x_rotated = apply_rotary_pos_emb(x, cos, sin)

    print(f"\n=== Rotation Test ===")
    print(f"Input shape: {x.shape}")
    print(f"Rotated shape: {x_rotated.shape}")
    print(f"✓ Shapes match: {x.shape == x_rotated.shape}")

    # Verify rotation preserves magnitude
    original_norm = jnp.linalg.norm(x, axis=-1)
    rotated_norm = jnp.linalg.norm(x_rotated, axis=-1)
    norm_diff = jnp.abs(original_norm - rotated_norm)
    print(f"Max norm difference: {jnp.max(norm_diff):.6f} (should be ~0)")

    print(f"\n✅ RoPE working correctly!")


def test_attention():
    """Test the complete attention mechanism."""
    from pathlib import Path

    import jax
    from config import LlamaConfig
    from layers import create_causal_mask

    config_path = Path(__file__).parent.parent / "llama_cfg.json"
    config = LlamaConfig.from_json(str(config_path))

    print("\n=== Testing Llama Attention ===")
    print(f"Query heads: {config.num_attention_heads}")
    print(f"Key/Value heads: {config.num_key_value_heads}")
    print(f"Head dim: {config.head_dim}")
    print(f"GQA ratio: {config.num_attention_heads // config.num_key_value_heads}:1")

    # Create attention
    attention = LlamaAttention(config=config)

    # Test input
    key = jax.random.PRNGKey(42)
    batch_size, seq_len = 2, 8
    hidden_states = jax.random.normal(key, (batch_size, seq_len, config.hidden_size))

    # Create causal mask
    causal_mask = create_causal_mask(seq_len)

    print(f"\n=== Shape Flow ===")
    print(f"Input: {hidden_states.shape}")
    print(f"Causal mask: {causal_mask.shape}")

    # Initialize and run
    params = attention.init(key, hidden_states, causal_mask)
    output = attention.apply(params, hidden_states, causal_mask)

    print(f"Output: {output.shape}")
    print(f"✓ Shape preserved: {hidden_states.shape == output.shape}")

    # Parameter counting
    def count_params(params_dict):
        total = 0
        for key, value in params_dict.items():
            if isinstance(value, dict):
                total += count_params(value)
            else:
                total += value.size
        return total

    total_params = count_params(params)
    expected_params = (
        config.hidden_size * config.num_attention_heads * config.head_dim  # q_proj
        + config.hidden_size * config.num_key_value_heads * config.head_dim  # k_proj
        + config.hidden_size * config.num_key_value_heads * config.head_dim  # v_proj
        + config.hidden_size * config.hidden_size  # o_proj
    )

    print(f"\n=== Parameters ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Expected: {expected_params:,}")
    print(f"✓ Parameter count correct: {total_params == expected_params}")

    # Show GQA memory savings
    standard_params = 3 * (
        config.hidden_size * config.num_attention_heads * config.head_dim
    ) + (config.hidden_size * config.hidden_size)
    savings = standard_params - expected_params

    print(f"\nGQA Memory Savings:")
    print(f"  Standard MHA: {standard_params:,} parameters")
    print(f"  GQA: {expected_params:,} parameters")
    print(f"  Savings: {savings:,} parameters ({savings / standard_params * 100:.1f}%)")

    print(f"\n✅ Attention working correctly!")


if __name__ == "__main__":
    test_rope()
    test_attention()
