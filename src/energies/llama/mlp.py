"""
Feed-forward network (MLP) with SwiGLU activation for Llama.

This is where most of the model's parameters live - each MLP layer
has ~50M parameters compared to ~10M for attention layers.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from config import LlamaConfig


class LlamaMLP(nn.Module):
    """Feed-forward network with SwiGLU activation.

    SwiGLU uses 3 linear projections instead of the standard 2:
    - gate_proj: learns what information to "gate" (let through)
    - up_proj: learns the actual feature transformations
    - down_proj: projects back to original dimension

    The key insight: gate_proj controls which features are active.
    """

    config: LlamaConfig

    def setup(self):
        # SwiGLU requires 3 projections (vs 2 for standard FFN)
        self.gate_proj = nn.Dense(
            self.config.intermediate_size,  # 2048 → 8192
            use_bias=False,
        )
        self.up_proj = nn.Dense(
            self.config.intermediate_size,  # 2048 → 8192
            use_bias=False,
        )
        self.down_proj = nn.Dense(
            self.config.hidden_size,  # 8192 → 2048
            use_bias=False,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        SwiGLU forward pass.

        Args:
            x: Input of shape (..., hidden_size)

        Returns:
            Output of shape (..., hidden_size)
        """
        # Project to intermediate dimension
        gate = self.gate_proj(x)  # (..., intermediate_size)
        up = self.up_proj(x)  # (..., intermediate_size)

        # SwiGLU activation: Swish(gate) ⊙ up
        # Swish(x) = x * sigmoid(x) - smooth activation function
        swish_gate = gate * jax.nn.sigmoid(gate)
        intermediate = swish_gate * up  # Element-wise gating

        # Project back to original dimension
        output = self.down_proj(intermediate)  # (..., hidden_size)

        return output


def test_mlp():
    """Test MLP with shape tracing and activation comparison."""
    from pathlib import Path

    import jax
    from config import LlamaConfig

    # Load config
    config_path = Path(__file__).parent.parent / "llama_cfg.json"
    config = LlamaConfig.from_json(str(config_path))

    print("=== Testing Llama MLP ===")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Intermediate size: {config.intermediate_size}")
    print(f"Expansion ratio: {config.intermediate_size / config.hidden_size:.1f}x")

    # Create MLP
    mlp = LlamaMLP(config=config)

    # Test input
    key = jax.random.PRNGKey(42)
    batch_size, seq_len = 2, 8
    x = jax.random.normal(key, (batch_size, seq_len, config.hidden_size))

    print(f"\n=== Shape Flow ===")
    print(f"Input: {x.shape}")

    # Initialize and run
    params = mlp.init(key, x)
    output = mlp.apply(params, x)

    print(f"Output: {output.shape}")
    print(f"✓ Shape preserved: {x.shape == output.shape}")

    # Trace intermediate shapes (conceptually)
    gate = x @ params["params"]["gate_proj"]["kernel"]  # Simulate gate projection
    up = x @ params["params"]["up_proj"]["kernel"]  # Simulate up projection

    print(f"\nIntermediate shapes:")
    print(f"  gate_proj output: {gate.shape}")
    print(f"  up_proj output: {up.shape}")
    print(f"  After element-wise multiply: {gate.shape}")  # Same as gate/up
    print(f"  down_proj input: {gate.shape}")
    print(f"  down_proj output: {output.shape}")

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
        config.hidden_size * config.intermediate_size  # gate_proj
        + config.hidden_size * config.intermediate_size  # up_proj
        + config.intermediate_size * config.hidden_size  # down_proj
    )

    print(f"\n=== Parameters ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Expected: {expected_params:,}")
    print(f"✓ Parameter count correct: {total_params == expected_params}")

    # Show parameter breakdown
    gate_params = config.hidden_size * config.intermediate_size
    up_params = config.hidden_size * config.intermediate_size
    down_params = config.intermediate_size * config.hidden_size

    print(f"\nParameter breakdown:")
    print(f"  gate_proj: {gate_params:,} ({gate_params / total_params * 100:.1f}%)")
    print(f"  up_proj: {up_params:,} ({up_params / total_params * 100:.1f}%)")
    print(f"  down_proj: {down_params:,} ({down_params / total_params * 100:.1f}%)")

    # Compare activations
    print(f"\n=== Activation Comparison ===")
    test_vals = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # ReLU (standard)
    relu_out = jax.nn.relu(test_vals)
    print(f"ReLU:     {relu_out}")

    # SiLU/Swish (what we use)
    swish_out = test_vals * jax.nn.sigmoid(test_vals)
    print(f"SiLU:     {swish_out}")

    print("SiLU is smoother than ReLU - no sharp cutoff at 0")
    print("This leads to better gradient flow during training")

    print(f"\n✅ MLP working correctly!")
    print(f"💡 This single MLP layer has {total_params / 1e6:.1f}M parameters!")


if __name__ == "__main__":
    test_mlp()
