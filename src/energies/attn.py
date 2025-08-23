import math

import jax
import jax.numpy as jnp
from jax import random

# ============================================================================
# PYTORCH-STYLE ATTENTION HEAD (for comparison)
# ============================================================================

"""
# This is what it would look like in PyTorch:

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHead(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False) 
        self.W_v = nn.Linear(d_model, d_k, bias=False)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        Q = self.W_q(x)  # (batch_size, seq_len, d_k)
        K = self.W_k(x)  # (batch_size, seq_len, d_k)
        V = self.W_v(x)  # (batch_size, seq_len, d_k)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

# Usage:
# attention = AttentionHead(d_model=512, d_k=64)
# output, weights = attention(x)
"""

# ============================================================================
# JAX FUNCTIONAL ATTENTION HEAD
# ============================================================================


def init_attention_params(key, d_model, d_k):
    """Initialize parameters for a single attention head."""
    keys = random.split(key, 3)

    # Xavier initialization
    scale = math.sqrt(1.0 / d_model)

    params = {
        "W_q": random.normal(keys[0], (d_model, d_k)) * scale,
        "W_k": random.normal(keys[1], (d_model, d_k)) * scale,
        "W_v": random.normal(keys[2], (d_model, d_k)) * scale,
    }
    return params


def attention_head_forward(params, x):
    """
    Single attention head forward pass.

    Args:
        params: Dictionary with W_q, W_k, W_v matrices
        x: Input tensor of shape (batch_size, seq_len, d_model)

    Returns:
        output: Attention output (batch_size, seq_len, d_k)
        attention_weights: Attention weights (batch_size, seq_len, seq_len)
    """
    # Linear projections to get Q, K, V
    Q = jnp.dot(x, params["W_q"])  # (batch_size, seq_len, d_k)
    K = jnp.dot(x, params["W_k"])  # (batch_size, seq_len, d_k)
    V = jnp.dot(x, params["W_v"])  # (batch_size, seq_len, d_k)

    # Get dimensions
    d_k = Q.shape[-1]

    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = jnp.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(d_k)

    # Apply softmax to get attention weights
    attention_weights = jax.nn.softmax(scores, axis=-1)

    # Apply attention to values
    output = jnp.matmul(attention_weights, V)

    return output, attention_weights


# ============================================================================
# MULTI-HEAD ATTENTION (both styles)
# ============================================================================

"""
# PyTorch Multi-Head Attention:

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        
        self.heads = nn.ModuleList([
            AttentionHead(d_model, self.d_k) for _ in range(n_heads)
        ])
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # Run each head
        head_outputs = []
        head_weights = []
        for head in self.heads:
            output, weights = head(x)
            head_outputs.append(output)
            head_weights.append(weights)
        
        # Concatenate heads
        concat_output = torch.cat(head_outputs, dim=-1)
        
        # Final linear projection
        output = self.W_o(concat_output)
        return output, head_weights
"""


def init_multihead_attention_params(key, d_model, n_heads):
    """Initialize parameters for multi-head attention."""
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

    d_k = d_model // n_heads
    keys = random.split(key, n_heads + 1)

    # Initialize each attention head
    head_params = []
    for i in range(n_heads):
        head_param = init_attention_params(keys[i], d_model, d_k)
        head_params.append(head_param)

    # Output projection matrix
    scale = math.sqrt(1.0 / d_model)
    W_o = random.normal(keys[-1], (d_model, d_model)) * scale

    return {"heads": head_params, "W_o": W_o}


def multihead_attention_forward(params, x):
    """
    Multi-head attention forward pass.

    Args:
        params: Dictionary with 'heads' (list of head params) and 'W_o'
        x: Input tensor (batch_size, seq_len, d_model)

    Returns:
        output: Final output (batch_size, seq_len, d_model)
        all_attention_weights: List of attention weights from each head
    """
    head_outputs = []
    all_attention_weights = []

    # Process each head
    for head_params in params["heads"]:
        head_output, head_weights = attention_head_forward(head_params, x)
        head_outputs.append(head_output)
        all_attention_weights.append(head_weights)

    # Concatenate all head outputs along the last dimension
    concat_output = jnp.concatenate(head_outputs, axis=-1)

    # Final linear projection
    output = jnp.dot(concat_output, params["W_o"])

    return output, all_attention_weights


# ============================================================================
# DEMONSTRATION AND COMPARISON
# ============================================================================


def demonstrate_attention():
    """Demonstrate attention mechanisms with detailed explanations."""
    print("=" * 70)
    print("ATTENTION MECHANISM COMPARISON: PyTorch vs JAX")
    print("=" * 70)

    # Setup
    key = random.PRNGKey(42)
    batch_size, seq_len, d_model = 2, 8, 512
    n_heads = 8

    # Create sample input (representing token embeddings)
    x = random.normal(key, (batch_size, seq_len, d_model))

    print(f"Input shape: {x.shape}")
    print(f"Model dimension: {d_model}")
    print(f"Number of heads: {n_heads}")
    print(f"Head dimension: {d_model // n_heads}")

    # ========================================================================
    # Single Head Attention
    # ========================================================================
    print("\n" + "-" * 50)
    print("SINGLE HEAD ATTENTION")
    print("-" * 50)

    single_head_key, multi_head_key = random.split(key)
    d_k = d_model // n_heads  # 64

    # Initialize single head
    single_head_params = init_attention_params(single_head_key, d_model, d_k)

    print(f"\nParameter shapes:")
    for name, param in single_head_params.items():
        print(f"  {name}: {param.shape}")

    # Forward pass
    single_output, single_weights = attention_head_forward(single_head_params, x)

    print(f"\nOutput shapes:")
    print(f"  Attention output: {single_output.shape}")
    print(f"  Attention weights: {single_weights.shape}")

    # Analyze attention pattern
    print(f"\nAttention analysis:")
    print(
        f"  Attention weights sum (should be ~1.0): {single_weights.sum(axis=-1).mean():.6f}"
    )
    print(f"  Max attention weight: {single_weights.max():.6f}")
    print(f"  Min attention weight: {single_weights.min():.6f}")

    # ========================================================================
    # Multi-Head Attention
    # ========================================================================
    print("\n" + "-" * 50)
    print("MULTI-HEAD ATTENTION")
    print("-" * 50)

    # Initialize multi-head attention
    mha_params = init_multihead_attention_params(multi_head_key, d_model, n_heads)

    print(f"\nParameter structure:")
    print(f"  Number of heads: {len(mha_params['heads'])}")
    print(f"  Output projection W_o: {mha_params['W_o'].shape}")

    # Forward pass
    mha_output, all_head_weights = multihead_attention_forward(mha_params, x)

    print(f"\nOutput shapes:")
    print(f"  Final output: {mha_output.shape}")
    print(f"  Number of attention weight matrices: {len(all_head_weights)}")
    print(f"  Each attention weight matrix: {all_head_weights[0].shape}")

    # ========================================================================
    # Key Differences Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("KEY DIFFERENCES: PyTorch vs JAX")
    print("=" * 70)

    print("\n🏗️  ARCHITECTURE:")
    print("  PyTorch: Object-oriented with classes and methods")
    print("           - State stored in self.parameters()")
    print("           - Forward pass via self.forward()")
    print("           - Automatic parameter management")

    print("\n  JAX:     Functional with explicit parameter passing")
    print("           - Parameters are plain dictionaries/arrays")
    print("           - Functions take (params, inputs) -> outputs")
    print("           - Manual parameter management")

    print("\n🔧  PARAMETER HANDLING:")
    print("  PyTorch: model.parameters() automatically tracks everything")
    print("  JAX:     You explicitly pass parameter dictionaries around")

    print("\n⚡  TRANSFORMATIONS:")
    print("  PyTorch: Limited built-in transformations")
    print("  JAX:     Rich transformation ecosystem (jit, vmap, grad, etc.)")

    # Demonstrate JAX transformations
    print("\n🚀  JAX TRANSFORMATIONS DEMO:")

    # JIT compile the attention
    compiled_attention = jax.jit(multihead_attention_forward)

    # Vectorize over different inputs (batch of batches)
    def process_batch(single_x):
        return multihead_attention_forward(mha_params, single_x)[0]

    vectorized_attention = jax.vmap(process_batch)

    # Create multiple batches
    multiple_batches = random.normal(key, (3, batch_size, seq_len, d_model))
    vectorized_output = vectorized_attention(multiple_batches)

    print(f"  Original input: {x.shape}")
    print(f"  Multiple batches: {multiple_batches.shape}")
    print(f"  Vectorized output: {vectorized_output.shape}")

    # Gradient computation
    def attention_loss(params, x, target):
        output, _ = multihead_attention_forward(params, x)
        return jnp.mean((output - target) ** 2)

    target = random.normal(key, mha_output.shape)
    grad_fn = jax.grad(attention_loss)
    gradients = grad_fn(mha_params, x, target)

    print(f"  Computed gradients for all parameters automatically!")
    print(f"  Gradient keys: {list(gradients.keys())}")

    return mha_params, x, mha_output, all_head_weights


def visualize_attention_pattern(attention_weights, head_idx=0):
    """Simple text visualization of attention patterns."""
    print(f"\n📊 ATTENTION PATTERN VISUALIZATION (Head {head_idx}):")
    print("   " + "".join([f"{i:4d}" for i in range(attention_weights.shape[-1])]))

    weights = attention_weights[0, :, :]  # First batch, all positions

    for i, row in enumerate(weights):
        # Convert to simple text visualization
        viz = ""
        for weight in row:
            if weight > 0.3:
                viz += " ██ "
            elif weight > 0.1:
                viz += " ▓▓ "
            elif weight > 0.05:
                viz += " ░░ "
            else:
                viz += "    "
        print(f"{i:2d}:{viz}")

    print("   Legend: ██ high attention, ▓▓ medium, ░░ low, (empty) very low")


if __name__ == "__main__":
    # Run the demonstration
    params, input_x, output, head_weights = demonstrate_attention()

    # Visualize attention patterns
    visualize_attention_pattern(head_weights[0])  # First head
    visualize_attention_pattern(head_weights[-1], head_idx=7)  # Last head

    print("\n" + "=" * 70)
    print("SUMMARY: Ready for Transformer Blocks!")
    print("=" * 70)
    print("✓ Single-head attention: Q, K, V projections + scaled dot-product")
    print("✓ Multi-head attention: Parallel heads + concatenation + output projection")
    print("✓ JAX functional style: Explicit parameter passing")
    print("✓ JAX transformations: jit, vmap, grad work seamlessly")
    print("✓ Attention visualization: See which tokens attend to which")
    print(
        "\nNext: Layer normalization, residual connections, and full transformer blocks!"
    )
