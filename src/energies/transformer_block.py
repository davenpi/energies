"""
Complete Transformer Block: What Happens After Attention

This module shows the full transformer block pipeline:
1. Multi-Head Attention
2. Add & Norm (residual connection + layer normalization)
3. Feed-Forward Network (MLP)
4. Add & Norm (residual connection + layer normalization)
"""

import math

import jax
import jax.numpy as jnp
from jax import random

# ============================================================================
# LAYER NORMALIZATION
# ============================================================================


def init_layer_norm_params(d_model):
    """Initialize layer normalization parameters."""
    return {
        "gamma": jnp.ones(d_model),  # Scale parameter
        "beta": jnp.zeros(d_model),  # Shift parameter
    }


def layer_norm(params, x, eps=1e-6):
    """
    Apply layer normalization.

    Args:
        params: Dict with 'gamma' and 'beta' parameters
        x: Input tensor (..., d_model)
        eps: Small constant for numerical stability

    Returns:
        normalized: Layer normalized tensor
    """
    # Compute mean and variance along the last dimension (d_model)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)

    # Normalize
    normalized = (x - mean) / jnp.sqrt(var + eps)

    # Scale and shift
    return params["gamma"] * normalized + params["beta"]


# ============================================================================
# FEED-FORWARD NETWORK
# ============================================================================


def init_ffn_params(key, d_model, d_ff):
    """
    Initialize feed-forward network parameters.

    Args:
        key: Random key
        d_model: Model dimension
        d_ff: Feed-forward dimension (typically 4 * d_model)

    Returns:
        params: Dict with W1, b1, W2, b2
    """
    key1, key2 = random.split(key)

    # Xavier initialization
    scale1 = math.sqrt(1.0 / d_model)
    scale2 = math.sqrt(1.0 / d_ff)

    return {
        "W1": random.normal(key1, (d_model, d_ff)) * scale1,
        "b1": jnp.zeros(d_ff),
        "W2": random.normal(key2, (d_ff, d_model)) * scale2,
        "b2": jnp.zeros(d_model),
    }


def feed_forward(params, x):
    """
    Feed-forward network: FFN(x) = max(0, xW1 + b1)W2 + b2

    Args:
        params: FFN parameters
        x: Input tensor (..., d_model)

    Returns:
        output: FFN output (..., d_model)
    """
    # First linear layer + ReLU
    hidden = jax.nn.relu(jnp.dot(x, params["W1"]) + params["b1"])

    # Second linear layer
    output = jnp.dot(hidden, params["W2"]) + params["b2"]

    return output


# ============================================================================
# MULTI-HEAD ATTENTION (from previous modules)
# ============================================================================


def init_attention_params(key, d_model, d_k):
    """Initialize single attention head parameters."""
    keys = random.split(key, 3)
    scale = math.sqrt(1.0 / d_model)

    return {
        "W_q": random.normal(keys[0], (d_model, d_k)) * scale,
        "W_k": random.normal(keys[1], (d_model, d_k)) * scale,
        "W_v": random.normal(keys[2], (d_model, d_k)) * scale,
    }


def create_causal_mask(seq_len):
    """Create causal mask for autoregressive attention."""
    return jnp.tril(jnp.ones((seq_len, seq_len))).astype(bool)


def attention_head_forward(params, x, causal_mask=None):
    """Single attention head with optional causal masking."""
    Q = jnp.dot(x, params["W_q"])
    K = jnp.dot(x, params["W_k"])
    V = jnp.dot(x, params["W_v"])

    d_k = Q.shape[-1]
    scores = jnp.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(d_k)

    # Apply causal mask if provided
    if causal_mask is not None:
        scores = jnp.where(causal_mask, scores, -1e9)

    weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(weights, V)

    return output, weights


def init_multihead_attention_params(key, d_model, n_heads):
    """Initialize multi-head attention parameters."""
    assert d_model % n_heads == 0
    d_k = d_model // n_heads

    keys = random.split(key, n_heads + 1)

    # Initialize each head
    head_params = []
    for i in range(n_heads):
        head_params.append(init_attention_params(keys[i], d_model, d_k))

    # Output projection
    scale = math.sqrt(1.0 / d_model)
    W_o = random.normal(keys[-1], (d_model, d_model)) * scale

    return {"heads": head_params, "W_o": W_o}


def multihead_attention_forward(params, x, causal_mask=None):
    """Multi-head attention forward pass."""
    head_outputs = []

    # Process each head
    for head_params in params["heads"]:
        head_output, _ = attention_head_forward(head_params, x, causal_mask)
        head_outputs.append(head_output)

    # Concatenate heads
    concat_output = jnp.concatenate(head_outputs, axis=-1)

    # Final projection
    output = jnp.dot(concat_output, params["W_o"])

    return output


# ============================================================================
# COMPLETE TRANSFORMER BLOCK
# ============================================================================


def init_transformer_block_params(key, d_model, n_heads, d_ff):
    """
    Initialize all parameters for a complete transformer block.

    Args:
        key: Random key
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension

    Returns:
        params: Complete transformer block parameters
    """
    key_attn, key_ffn = random.split(key)

    return {
        "attention": init_multihead_attention_params(key_attn, d_model, n_heads),
        "ln1": init_layer_norm_params(d_model),  # Layer norm after attention
        "ffn": init_ffn_params(key_ffn, d_model, d_ff),
        "ln2": init_layer_norm_params(d_model),  # Layer norm after FFN
    }


def transformer_block_forward(params, x, causal_mask=None):
    """
    Complete transformer block forward pass.

    The standard transformer block follows this pattern:
    1. Multi-head attention
    2. Add & Norm (residual connection + layer normalization)
    3. Feed-forward network
    4. Add & Norm (residual connection + layer normalization)

    Args:
        params: Transformer block parameters
        x: Input tensor (batch_size, seq_len, d_model)
        causal_mask: Optional causal mask for attention

    Returns:
        output: Transformer block output (batch_size, seq_len, d_model)
    """
    # 1. Multi-head attention
    attn_output = multihead_attention_forward(params["attention"], x, causal_mask)

    # 2. Add & Norm (residual connection + layer normalization)
    x = layer_norm(params["ln1"], x + attn_output)

    # 3. Feed-forward network
    ffn_output = feed_forward(params["ffn"], x)

    # 4. Add & Norm (residual connection + layer normalization)
    output = layer_norm(params["ln2"], x + ffn_output)

    return output


# ============================================================================
# DEMONSTRATION: STEP BY STEP
# ============================================================================


def demonstrate_transformer_block():
    """Walk through each step of the transformer block."""
    print("=" * 80)
    print("COMPLETE TRANSFORMER BLOCK: STEP BY STEP")
    print("=" * 80)

    # Configuration
    key = random.PRNGKey(42)
    batch_size, seq_len, d_model = 2, 8, 64
    n_heads = 8
    d_ff = 256  # Typically 4 * d_model

    # Create input (embeddings + positional encoding)
    x = random.normal(key, (batch_size, seq_len, d_model))

    print(f"Configuration:")
    print(f"  Input shape: {x.shape}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Feed-forward dimension: {d_ff}")

    # Initialize transformer block
    block_params = init_transformer_block_params(key, d_model, n_heads, d_ff)

    # Create causal mask for autoregressive generation
    causal_mask = create_causal_mask(seq_len)

    print(f"\nParameter counts:")
    # Count parameters in each component
    attn_params = n_heads * 3 * d_model * (d_model // n_heads) + d_model * d_model
    ffn_params = d_model * d_ff + d_ff + d_ff * d_model + d_model
    ln_params = 2 * (d_model + d_model)  # 2 layer norms, each with gamma + beta
    total_params = attn_params + ffn_params + ln_params

    print(f"  Attention parameters: {attn_params:,}")
    print(f"  Feed-forward parameters: {ffn_params:,}")
    print(f"  Layer norm parameters: {ln_params:,}")
    print(f"  Total parameters: {total_params:,}")

    # ========================================================================
    # Step-by-step forward pass
    # ========================================================================
    print(f"\n{'=' * 60}")
    print("STEP-BY-STEP FORWARD PASS")
    print("=" * 60)

    print(f"\n1️⃣ INPUT:")
    print(f"   Shape: {x.shape}")
    print(f"   This represents {batch_size} sequences of {seq_len} tokens")
    print(f"   Each token is a {d_model}-dimensional vector")

    # Step 1: Multi-head attention
    attn_output = multihead_attention_forward(block_params["attention"], x, causal_mask)
    print(f"\n2️⃣ MULTI-HEAD ATTENTION:")
    print(f"   Input:  {x.shape}")
    print(f"   Output: {attn_output.shape}")
    print(f"   What it does: Each token attends to previous tokens")
    print(f"   Causal mask: Prevents attending to future tokens")

    # Step 2: Add & Norm 1
    after_attn_residual = x + attn_output
    after_ln1 = layer_norm(block_params["ln1"], after_attn_residual)
    print(f"\n3️⃣ ADD & NORM 1:")
    print(f"   Residual: {x.shape} + {attn_output.shape} = {after_attn_residual.shape}")
    print(f"   Layer norm: {after_ln1.shape}")
    print(f"   What it does: Stabilizes training, enables deep networks")

    # Step 3: Feed-forward network
    ffn_output = feed_forward(block_params["ffn"], after_ln1)
    print(f"\n4️⃣ FEED-FORWARD NETWORK:")
    print(f"   Input:  {after_ln1.shape}")
    print(f"   Hidden: {(batch_size, seq_len, d_ff)} (after first layer)")
    print(f"   Output: {ffn_output.shape}")
    print(f"   What it does: Non-linear transformation, adds capacity")

    # Step 4: Add & Norm 2
    after_ffn_residual = after_ln1 + ffn_output
    final_output = layer_norm(block_params["ln2"], after_ffn_residual)
    print(f"\n5️⃣ ADD & NORM 2:")
    print(
        f"   Residual: {after_ln1.shape} + {ffn_output.shape} = {after_ffn_residual.shape}"
    )
    print(f"   Final output: {final_output.shape}")
    print(f"   What it does: Final stabilization")

    # Verify with single function call
    direct_output = transformer_block_forward(block_params, x, causal_mask)
    print(f"\n✅ VERIFICATION:")
    print(
        f"   Step-by-step matches direct call: {jnp.allclose(final_output, direct_output)}"
    )

    return block_params, x, final_output


def explain_each_component():
    """Explain the purpose of each component in detail."""
    print(f"\n{'=' * 80}")
    print("WHY EACH COMPONENT IS NECESSARY")
    print("=" * 80)

    print(f"\n🎯 MULTI-HEAD ATTENTION:")
    print(f"   Purpose: Learn relationships between tokens")
    print(f"   Why multiple heads: Different types of relationships")
    print(f"     - Head 1: Syntactic dependencies (noun-verb)")
    print(f"     - Head 2: Semantic similarity (related concepts)")
    print(f"     - Head 3: Long-range dependencies")
    print(f"   Output: Context-aware representations")

    print(f"\n🔗 RESIDUAL CONNECTIONS (Add):")
    print(f"   Purpose: Enable training of very deep networks")
    print(f"   Problem solved: Vanishing gradients")
    print(f"   How: x_out = x_in + f(x_in)")
    print(f"   Benefit: Gradients can flow directly through shortcuts")

    print(f"\n📏 LAYER NORMALIZATION:")
    print(f"   Purpose: Stabilize training and speed up convergence")
    print(f"   What it does: Normalize activations to have mean=0, std=1")
    print(f"   Where: After each major component (attention, FFN)")
    print(f"   Learnable: Yes (gamma for scale, beta for shift)")

    print(f"\n🧠 FEED-FORWARD NETWORK:")
    print(f"   Purpose: Add non-linear processing capacity")
    print(f"   Structure: Linear -> ReLU -> Linear")
    print(f"   Dimension: d_model -> d_ff -> d_model (d_ff ≈ 4 * d_model)")
    print(f"   Why needed: Attention is mostly linear operations")

    print(f"\n🏗️ THE COMPLETE ARCHITECTURE:")
    print(f"   Input -> [Attention -> Add&Norm -> FFN -> Add&Norm] -> Output")
    print(f"   This block can be stacked N times (N=12 for GPT-1, N=96 for GPT-3)")
    print(f"   Each block has the same structure but different parameters")


def common_variations():
    """Show common variations of the transformer block."""
    print(f"\n{'=' * 80}")
    print("COMMON TRANSFORMER BLOCK VARIATIONS")
    print("=" * 80)

    print(f"\n📚 ORIGINAL TRANSFORMER (Vaswani et al.):")
    print(f"   Structure: Attention -> Add&Norm -> FFN -> Add&Norm")
    print(f"   Layer norm: After the residual connection (Post-LN)")

    print(f"\n🤖 GPT STYLE (Pre-LN):")
    print(f"   Structure: LN -> Attention -> Add -> LN -> FFN -> Add")
    print(f"   Layer norm: Before the main operation (Pre-LN)")
    print(f"   Benefit: More stable training for very deep networks")

    print(f"\n🔄 T5 STYLE:")
    print(f"   Structure: Similar to Pre-LN but with RMSNorm instead of LayerNorm")
    print(f"   RMSNorm: Only normalizes by RMS, no mean centering")

    print(f"\n⚡ EFFICIENCY VARIATIONS:")
    print(f"   - Linear attention: O(n) instead of O(n²)")
    print(f"   - Sparse attention: Only attend to subset of positions")
    print(f"   - Sliding window: Only attend to nearby positions")
    print(f"   - Flash attention: Memory-efficient attention computation")


if __name__ == "__main__":
    # Demonstrate complete transformer block
    params, input_x, output = demonstrate_transformer_block()

    # Explain each component
    explain_each_component()

    # Show variations
    common_variations()

    print(f"\n{'=' * 80}")
    print("🎉 TRANSFORMER BLOCK: COMPLETE!")
    print("=" * 80)
    print("✅ Multi-head attention: Context-aware representations")
    print("✅ Residual connections: Enable deep networks")
    print("✅ Layer normalization: Stable training")
    print("✅ Feed-forward network: Non-linear processing")
    print("✅ The complete block can be stacked N times")
    print("\n🚀 Ready to build a full transformer model!")
