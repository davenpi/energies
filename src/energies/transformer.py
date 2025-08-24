"""
Transformer architecture.
"""

import jax
import jax.numpy as jnp

# ============= Constants =============
vocab_size = 10000
batch_size = 1
max_seq_len = 10
d_model = 128
d_k = 64
d_ff = 4 * d_model
num_heads = d_model // d_k


# ============= Embeddings =============
def init_embeddings_params(key: int, vocab_size: int, d_model: int, max_seq_len: int):
    """Initialize embeddings parameters."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(key))
    return {
        "W_e": jax.random.normal(k1, (vocab_size, d_model)),
        "W_pos": jax.random.normal(k2, (max_seq_len, d_model)),
    }


def embed_forward(params: dict, x: jnp.ndarray) -> jax.Array:
    """Do embedding lookup and add positional encodings.

    Args:
        params: Dictionary with 'W_e' and 'W_pos' parameters
        x: Token IDs (..., seq_len)

    Returns:
        embeddings: Embeddings (..., seq_len, d_model)
    """
    # Embed tokens
    embeddings: jax.Array = params["W_e"][x]

    # Positional encodings
    pos_enc: jax.Array = params["W_pos"][: x.shape[-1]]

    return embeddings + pos_enc


# ============= Attention =============


def init_attention_head_params(
    key: int, d_model: int, d_k: int
) -> dict[str, jax.Array]:
    """Initialize attention parameters."""
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(key))
    return {
        "W_q": jax.random.normal(k1, (d_model, d_k)),
        "W_k": jax.random.normal(k2, (d_model, d_k)),
        "W_v": jax.random.normal(k3, (d_model, d_k)),
    }


def causal_mask(seq_len: int) -> jnp.Array:
    """Create a causal mask.

    Args:
        seq_len: Sequence length

    Returns:
        mask: Causal mask (seq_len, seq_len)
    """
    return jnp.tril(jnp.ones((seq_len, seq_len)))


def attn_head(params: dict, causal_mask: jnp.Array, x: jnp.ndarray) -> jax.Array:
    """Compute scaled dot product attention

    Args:
        params: Dictionary with `W_q`, `W_k`, and `W_v`
        x: Input data (batch_size, seq_len, d_model)

    Output:
        Scaled dot product attention.
    """
    # Extract params
    W_q = params["W_q"]
    W_k = params["W_k"]
    W_v = params["W_v"]

    # Compute projections
    Q = jnp.dot(x, W_q)  # shape (batch, seq_len, d_k)
    K = jnp.dot(x, W_k)  # shape (batch, seq_len, d_k)
    V = jnp.dot(x, W_v)  # shape (batch, seq_len, d_k)

    # Attention weights
    attn_scores = jnp.dot(Q, K.transpose(0, 2, 1))  # shape (batch, seq_len, seq_len)

    # Mask out future tokens
    attn_scores = jnp.where(causal_mask, attn_scores, -jnp.inf)

    # Scale
    attn_scores /= jnp.sqrt(d_k)

    # Softmax
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)

    # SDPA
    return jnp.dot(attn_weights, V)  # shape (batch, seq_len, d_k)


def multi_head_attn(
    wo_params: dict, attn_params: list[dict], causal_mask: jnp.Array, x: jnp.ndarray
) -> jax.Array:
    """Compute attention across all heads and project.

    Args:
        wo_params: Dictionary with 'W_o' parameter
        attn_params: List of dictionaries with 'W_q', 'W_k', and 'W_v' parameters
        causal_mask: Causal mask (seq_len, seq_len)
        x: Input data (batch_size, seq_len, d_model)

    Returns:
        multi_head_attn: Multi-head attention (batch_size, seq_len, d_model)
    """
    attn_arrays = []
    w_o = wo_params["W_o"]

    # Compute attn heads
    for p in attn_params:
        attn_arrays.append(
            attn_head(p, causal_mask, x)
        )  # each shape (batch, seq_len, d_k)

    multi_head_attn = jnp.concatenate(
        attn_arrays, axis=-1
    )  # shape (batch, seq_len, d_model) since num_heads * d_k = d_model

    return jnp.dot(multi_head_attn, w_o)  # shape (batch, seq_len, d_model)


def init_ffn_params(key: int, d_model: int, d_ff: int) -> dict:
    """Initialize feed-forward network parameters."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(key))
    return {
        "W_1": jax.random.normal(k1, (d_model, d_ff)),
        "W_2": jax.random.normal(k2, (d_ff, d_model)),
    }


def feed_forward(params: dict, x: jnp.ndarray) -> jax.Array:
    """Compute feed-forward network.

    Args:
        params: Dictionary with 'W_1' and 'W_2' parameters
        x: Input data (batch_size, seq_len, d_model)

    Returns:
        feed_forward: Feed-forward network (batch_size, seq_len, d_model)
    """
    # Feed-forward network
    proj = jnp.dot(x, params["W_1"])  # shape (batch, seq_len, d_ff)

    # ReLU
    relu = jax.nn.relu(proj)

    # Output
    return jnp.dot(relu, params["W_2"])  # shape (batch, seq_len, d_model)


def init_layer_norm_params(key: int, d_model: int) -> dict:
    """Initialize layer normalization parameters for a transformer block."""
    return {
        "g1": jax.random.normal(jax.random.PRNGKey(key), (d_model,)),
        "b1": jax.random.normal(jax.random.PRNGKey(key), (d_model,)),
        "g2": jax.random.normal(jax.random.PRNGKey(key), (d_model,)),
        "b2": jax.random.normal(jax.random.PRNGKey(key), (d_model,)),
    }


def layer_norm(gamma: jnp.ndarray, beta: jnp.ndarray, x: jnp.ndarray) -> jax.Array:
    """Apply layer normalization."""
    # Compute mean and variance
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)

    # Normalize
    normalized = (x - mean) / jnp.sqrt(var + 1e-6)

    # Scale and shift
    return gamma * normalized + beta


# ============= Transformer Block =============


def init_transformer_block_params(d_model: int, d_k: int, num_heads: int) -> dict:
    """Build the parameters for a transformer block.

    Assume d_ff = 4 * d_model, and num_heads * d_k = d_model
    """

    # Head params
    head_params: list[dict] = []
    for i in range(num_heads):
        head_params.append(init_attention_head_params(i, d_model, d_k))

    # Wo params
    wo_params = jax.random.normal(jax.random.PRNGKey(0), (d_model, d_model))

    # First layer norm params
    layer_norm_params = {
        "g1": jnp.ones((d_model,)),
        "b1": jnp.zeros((d_model,)),
    }

    # FFN params
    ffn_params = {
        "W_1": jnp.zeros((d_model, 4 * d_model)),
        "W_2": jnp.zeros((4 * d_model, d_model)),
    }

    # Second layer norm params
    layer_norm_params = {
        "g2": jnp.ones((d_model,)),
        "b2": jnp.zeros((d_model,)),
    }

    return {
        "head_params": head_params,
        "wo_params": wo_params,
        "ffn_params": ffn_params,
        "layer_norm_params": layer_norm_params,
    }


def transformer_block(params: dict, x: jnp.ndarray) -> jax.Array:
    """Compute a transformer block.

    Args:
        params: Dictionary with params for attention heads, feed-forward network,
        and layer normalization.
        x: Input data (batch_size, seq_len, d_model)

    Returns:
        x: Output data (batch_size, seq_len, d_model)
    """
    # Step 1: Extract params
    head_params: list[dict] = params["head_params"]
    wo_params: dict = params["wo_params"]
    ffn_params: dict = params["ffn_params"]
    layer_norm_params: dict = params["layer_norm_params"]

    # Step 2: Attention
    mask = causal_mask(x.shape[1])
    attn_out = multi_head_attn(wo_params, head_params, mask, x)

    # Step 3: Residual connection
    x = x + attn_out

    # Step 4: Layer normalization
    x = layer_norm(layer_norm_params["g1"], layer_norm_params["b1"], x)

    # Step 5: Feed-forward
    ff_out = feed_forward(ffn_params, x)

    # Step 6: Residual connection
    x = x + ff_out

    # Step 7: Layer normalization
    x = layer_norm(layer_norm_params["g2"], layer_norm_params["b2"], x)

    return x


def transformer_fwd(
    embeddings_params: dict,
    transformer_blocks: list[dict],
    wo_params: dict,
    x: jnp.ndarray,
) -> jax.Array:
    # Step 1: Embeddings
    embeddings = embed_forward(embeddings_params, x)

    # Step 2: Transformer blocks
    for block in transformer_blocks:
        embeddings = transformer_block(block, embeddings)

    # Step 3: Output projection (batch_size, seq_len, vocab_size)
    output = jnp.dot(embeddings, wo_params["W_o"])

    return output
