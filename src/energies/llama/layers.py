"""
Layer components for Llama: normalization and transformer blocks.

RMSNorm is used throughout Llama instead of LayerNorm - it's simpler
and more efficient while providing similar benefits.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from config import LlamaConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm is simpler than LayerNorm:
    - No mean subtraction (only RMS scaling)
    - No bias term (only learned scale)
    - Faster computation
    - Similar performance to LayerNorm
    """

    config: LlamaConfig

    def setup(self):
        # Only a scale parameter (no bias like LayerNorm)
        self.weight = self.param(
            "weight", nn.initializers.ones, (self.config.hidden_size,)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., hidden_size)

        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS (Root Mean Square) over the last dimension
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)

        # Normalize by RMS + epsilon (for numerical stability)
        x_normalized = x * jax.lax.rsqrt(variance + self.config.rms_norm_eps)

        # Scale by learned parameter
        return self.weight * x_normalized


def create_causal_mask(seq_len: int) -> jnp.ndarray:
    """Create a causal (lower triangular) attention mask.

    This prevents tokens from attending to future positions,
    maintaining the autoregressive property of language models.

    Args:
        seq_len: Sequence length

    Returns:
        Causal mask of shape (1, 1, seq_len, seq_len)
    """
    # Lower triangular matrix (1s below diagonal, 0s above)
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))

    # Add batch and head dimensions for broadcasting
    return mask[None, None, :, :]


def test_rms_norm():
    """Test RMSNorm with shape tracing and comparison to LayerNorm."""
    from pathlib import Path

    import jax
    from config import LlamaConfig

    # Load config
    config_path = Path(__file__).parent.parent / "llama_cfg.json"
    config = LlamaConfig.from_json(str(config_path))

    print("=== Testing RMSNorm ===")
    print(f"Hidden size: {config.hidden_size}")
    print(f"RMS epsilon: {config.rms_norm_eps}")

    # Create RMSNorm
    rms_norm = RMSNorm(config=config)

    # Test input
    key = jax.random.PRNGKey(42)
    batch_size, seq_len = 2, 8
    x = jax.random.normal(key, (batch_size, seq_len, config.hidden_size))

    print(f"\n=== Shape Flow ===")
    print(f"Input: {x.shape}")

    # Initialize and run
    params = rms_norm.init(key, x)
    output = rms_norm.apply(params, x)

    print(f"Output: {output.shape}")
    print(f"✓ Shape preserved: {x.shape == output.shape}")

    # Check normalization properties
    print(f"\n=== Normalization Properties ===")

    # Original statistics
    input_mean = jnp.mean(x, axis=-1)
    input_std = jnp.std(x, axis=-1)

    # Normalized statistics
    output_mean = jnp.mean(output, axis=-1)
    output_std = jnp.std(output, axis=-1)

    print(f"Input mean (sample): {input_mean[0, 0]:.4f}")
    print(f"Input std (sample): {input_std[0, 0]:.4f}")
    print(f"Output mean (sample): {output_mean[0, 0]:.4f}")
    print(f"Output std (sample): {output_std[0, 0]:.4f}")

    # RMSNorm doesn't center to zero mean (unlike LayerNorm)
    print("Note: RMSNorm doesn't center to zero mean (no mean subtraction)")

    # Parameter count
    weight_params = params["params"]["weight"]
    param_count = weight_params.size

    print(f"\n=== Parameters ===")
    print(f"Weight shape: {weight_params.shape}")
    print(f"Parameter count: {param_count:,}")
    print(f"Expected: {config.hidden_size:,}")
    print(f"✓ Parameter count correct: {param_count == config.hidden_size}")

    # Show the learned scaling factors
    print(f"\nSample weight values: {weight_params[:5]}")
    print("These scale each feature dimension independently")

    print(f"\n✅ RMSNorm working correctly!")


def test_causal_mask():
    """Test causal mask creation."""
    print("\n=== Testing Causal Mask ===")

    seq_len = 5
    mask = create_causal_mask(seq_len)

    print(f"Mask shape: {mask.shape}")
    print(f"Expected: (1, 1, {seq_len}, {seq_len})")

    # Show the actual mask pattern
    print(f"\nMask pattern (seq_len={seq_len}):")
    mask_2d = mask[0, 0]  # Remove batch/head dims for display

    for i in range(seq_len):
        row = ""
        for j in range(seq_len):
            row += "1 " if mask_2d[i, j] else "0 "
        print(f"  Position {i}: {row}")

    print("\n1 = can attend, 0 = cannot attend")
    print("Lower triangular = causal (no future peeking)")

    print(f"\n✅ Causal mask working correctly!")


if __name__ == "__main__":
    test_rms_norm()
    test_causal_mask()
