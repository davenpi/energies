"""
Experimenting with multi-head attention.
"""

import jax
import jax.numpy as jnp

# ============= Constants =============

d_model = 128
seq_len = 16
d_k = 32
num_heads = d_model // d_k
print("num_heads:", num_heads)
print("d_k:", d_k)
print("d_model:", d_model)

# ============= Parameters =============


def init_attn_head_params(key: int, d_model: int, d_k: int) -> dict:
    """Initialize parameters for a multi-head attention block."""
    return {
        "W_q": jax.random.normal(jax.random.PRNGKey(key), (d_model, d_k)),
        "W_k": jax.random.normal(jax.random.PRNGKey(key), (d_model, d_k)),
        "W_v": jax.random.normal(jax.random.PRNGKey(key), (d_model, d_k)),
    }


def attn_head(params: dict, x: jnp.ndarray, causal_mask: jnp.ndarray) -> jnp.ndarray:
    """Compute attention head."""
    W_q = params["W_q"]
    W_k = params["W_k"]
    W_v = params["W_v"]

    Q = jnp.dot(x, W_q)  # (batch, seq_len, d_k)
    K = jnp.dot(x, W_k)  # (batch, seq_len, d_k)
    V = jnp.dot(x, W_v)  # (batch, seq_len, d_k)

    # Simple and clear: Q @ K.T
    attn_scores = jnp.matmul(Q, K.transpose(0, 2, 1))  # (batch, seq_len, seq_len)

    # Scale
    attn_scores = attn_scores / jnp.sqrt(d_k)

    # Mask
    attn_scores = jnp.where(causal_mask, attn_scores, -jnp.inf)

    # Softmax
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)

    # Apply to values
    return jnp.matmul(attn_weights, V)  # (batch, seq_len, d_k)


if __name__ == "__main__":
    # Step 1: Initialize params
    params = init_attn_head_params(0, d_model, d_k)

    # Step 2: Compute attention head
    x = jax.random.normal(jax.random.PRNGKey(0), (1, seq_len, d_model))
    print("x.shape:", x.shape)
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    print("causal_mask.shape:", causal_mask.shape)
    attn_out = attn_head(params, x, causal_mask)

    print("attn_out.shape:", attn_out.shape)
