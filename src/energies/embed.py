import math

import jax
import jax.numpy as jnp
from jax import random

# ============================================================================
# COMPLETE TRANSFORMER INPUT PIPELINE
# ============================================================================


def create_embedding_params(key, vocab_size, d_model):
    """Create token embedding parameters."""
    # Each token gets mapped to a d_model dimensional vector
    scale = math.sqrt(1.0 / d_model)
    embedding_matrix = random.normal(key, (vocab_size, d_model)) * scale
    return {"embedding_matrix": embedding_matrix}


def token_embedding(params, token_ids):
    """
    Convert token IDs to dense embeddings.

    Args:
        params: Dictionary with 'embedding_matrix' of shape (vocab_size, d_model)
        token_ids: Integer token IDs of shape (batch_size, seq_len)

    Returns:
        embeddings: Dense embeddings of shape (batch_size, seq_len, d_model)
    """
    # This is equivalent to one-hot encoding followed by matrix multiplication
    # but much more efficient
    return params["embedding_matrix"][token_ids]


def create_positional_encoding(seq_len, d_model):
    """
    Create sinusoidal positional encodings.

    Args:
        seq_len: Maximum sequence length
        d_model: Model dimension

    Returns:
        pos_encoding: Positional encodings of shape (seq_len, d_model)
    """
    position = jnp.arange(seq_len)[:, None]  # (seq_len, 1)
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

    pos_encoding = jnp.zeros((seq_len, d_model))
    pos_encoding = pos_encoding.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_encoding = pos_encoding.at[:, 1::2].set(jnp.cos(position * div_term))

    return pos_encoding


def add_positional_encoding(embeddings, pos_encoding):
    """Add positional encoding to token embeddings."""
    seq_len = embeddings.shape[1]
    return embeddings + pos_encoding[:seq_len]


# ============================================================================
# ATTENTION HEAD (from previous example)
# ============================================================================


def init_attention_params(key, d_model, d_k):
    """Initialize parameters for a single attention head."""
    keys = random.split(key, 3)
    scale = math.sqrt(1.0 / d_model)

    params = {
        "W_q": random.normal(keys[0], (d_model, d_k)) * scale,
        "W_k": random.normal(keys[1], (d_model, d_k)) * scale,
        "W_v": random.normal(keys[2], (d_model, d_k)) * scale,
    }
    return params


def attention_head_forward(params, x):
    """Single attention head forward pass."""
    Q = jnp.dot(x, params["W_q"])
    K = jnp.dot(x, params["W_k"])
    V = jnp.dot(x, params["W_v"])

    d_k = Q.shape[-1]
    scores = jnp.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(d_k)
    attention_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(attention_weights, V)

    return output, attention_weights


# ============================================================================
# COMPLETE PIPELINE DEMONSTRATION
# ============================================================================


def demonstrate_complete_pipeline():
    """Show the complete flow from tokens to attention."""
    print("=" * 80)
    print("COMPLETE TRANSFORMER INPUT PIPELINE")
    print("=" * 80)

    # Configuration
    vocab_size = 1000  # Size of our vocabulary
    d_model = 128  # Embedding/model dimension
    seq_len = 10  # Sequence length
    batch_size = 2  # Number of sequences in batch

    key = random.PRNGKey(42)
    embed_key, attn_key, token_key = random.split(key, 3)

    print(f"Configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension (d_model): {d_model}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")

    # ========================================================================
    # Step 1: Create sample token sequences
    # ========================================================================
    print(f"\n{'=' * 50}")
    print("STEP 1: RAW TOKEN SEQUENCES")
    print("=" * 50)

    # Generate random token IDs (in practice, these come from tokenizing text)
    token_ids = random.randint(token_key, (batch_size, seq_len), 0, vocab_size)

    print(f"Token IDs shape: {token_ids.shape}")
    print(f"Sample token IDs (first sequence): {token_ids[0]}")
    print(f"Token ID range: [{token_ids.min()}, {token_ids.max()}]")

    # ========================================================================
    # Step 2: Token Embeddings
    # ========================================================================
    print(f"\n{'=' * 50}")
    print("STEP 2: TOKEN EMBEDDINGS")
    print("=" * 50)

    # Create embedding parameters
    embed_params = create_embedding_params(embed_key, vocab_size, d_model)
    print(f"Embedding matrix shape: {embed_params['embedding_matrix'].shape}")

    # Convert tokens to embeddings
    token_embeddings = token_embedding(embed_params, token_ids)
    print(f"Token embeddings shape: {token_embeddings.shape}")
    print(f"Each token is now a {d_model}-dimensional vector")

    # Show what happened to first token
    first_token_id = token_ids[0, 0]
    first_token_embedding = token_embeddings[0, 0]
    print(
        f"\nExample: Token ID {first_token_id} -> embedding vector of length {len(first_token_embedding)}"
    )
    print(f"First few values: {first_token_embedding[:5]}")

    # ========================================================================
    # Step 3: Positional Encodings
    # ========================================================================
    print(f"\n{'=' * 50}")
    print("STEP 3: POSITIONAL ENCODINGS")
    print("=" * 50)

    pos_encoding = create_positional_encoding(seq_len, d_model)
    print(f"Positional encoding shape: {pos_encoding.shape}")

    # Add positional encodings to embeddings
    embeddings_with_pos = add_positional_encoding(token_embeddings, pos_encoding)
    print(f"Final embeddings shape: {embeddings_with_pos.shape}")

    # Show the difference
    print(f"\nPosition encoding effect on first token:")
    print(f"  Before: {token_embeddings[0, 0, :3]} ...")
    print(f"  After:  {embeddings_with_pos[0, 0, :3]} ...")
    print(f"  Pos encoding: {pos_encoding[0, :3]} ...")

    # ========================================================================
    # Step 4: Attention
    # ========================================================================
    print(f"\n{'=' * 50}")
    print("STEP 4: ATTENTION")
    print("=" * 50)

    # Initialize attention head
    d_k = d_model // 4  # Head dimension (could be d_model // num_heads)
    attn_params = init_attention_params(attn_key, d_model, d_k)

    print(f"Attention head configuration:")
    print(f"  Input dimension (d_model): {d_model}")
    print(f"  Head dimension (d_k): {d_k}")

    for name, param in attn_params.items():
        print(f"  {name} shape: {param.shape}")

    # Apply attention
    attn_output, attn_weights = attention_head_forward(attn_params, embeddings_with_pos)

    print(f"\nAttention results:")
    print(f"  Output shape: {attn_output.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")

    # ========================================================================
    # Analysis: What each dimension means
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("DIMENSION ANALYSIS")
    print("=" * 80)

    print(f"\n🔢 SHAPE EVOLUTION:")
    print(f"  Raw tokens:           {token_ids.shape} = (batch_size, seq_len)")
    print(
        f"  Token embeddings:     {token_embeddings.shape} = (batch_size, seq_len, d_model)"
    )
    print(
        f"  With positions:       {embeddings_with_pos.shape} = (batch_size, seq_len, d_model)"
    )
    print(f"  Attention output:     {attn_output.shape} = (batch_size, seq_len, d_k)")
    print(
        f"  Attention weights:    {attn_weights.shape} = (batch_size, seq_len, seq_len)"
    )

    print(f"\n📊 WHAT EACH DIMENSION MEANS:")
    print(
        f"  Batch dimension (B={batch_size}):     Different sequences being processed in parallel"
    )
    print(
        f"  Sequence dimension (N={seq_len}):     Positions in the sequence (tokens/words)"
    )
    print(
        f"  Model dimension (D={d_model}):        Feature dimension of each token representation"
    )
    print(
        f"  Head dimension (d_k={d_k}):           Feature dimension after projection in attention head"
    )

    print(f"\n🧠 ATTENTION WEIGHTS INTERPRETATION:")
    print(f"  Shape: (batch_size, seq_len, seq_len)")
    print(
        f"  attn_weights[b, i, j] = how much position i attends to position j in batch b"
    )
    print(f"  Each row sums to 1.0 (softmax normalization)")

    # Show attention pattern for first sequence
    print(f"\n📈 SAMPLE ATTENTION PATTERN (first sequence):")
    first_seq_weights = attn_weights[0]  # Shape: (seq_len, seq_len)

    print("     " + "".join([f"{j:6d}" for j in range(seq_len)]))
    for i in range(seq_len):
        row_str = f"Pos{i:2d}: "
        for j in range(seq_len):
            weight = first_seq_weights[i, j]
            row_str += f"{weight:6.3f}"
        print(row_str)

    print(f"\nRow sums (should all be ~1.0): {first_seq_weights.sum(axis=1)}")

    return {
        "token_ids": token_ids,
        "embeddings": embeddings_with_pos,
        "attention_output": attn_output,
        "attention_weights": attn_weights,
        "config": {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "seq_len": seq_len,
            "batch_size": batch_size,
        },
    }


def clarify_common_confusion():
    """Address common points of confusion."""
    print(f"\n{'=' * 80}")
    print("COMMON CONFUSION POINTS CLARIFIED")
    print("=" * 80)

    print(f"\n❓ CONFUSION: 'D is vocabulary size'")
    print(f"✅ REALITY: D is the model/embedding dimension")
    print(f"   - Vocabulary size: How many unique tokens exist (e.g., 50,000)")
    print(f"   - Model dimension: Size of dense vector for each token (e.g., 512)")
    print(f"   - We use an embedding lookup table: vocab_size × d_model")

    print(f"\n❓ CONFUSION: 'One-hot vectors as input'")
    print(f"✅ REALITY: Token IDs as input, embeddings via lookup")
    print(f"   - Input: [15, 42, 8] (token IDs)")
    print(f"   - NOT: [[0,0,1,0,...], [0,1,0,0,...], ...] (one-hot)")
    print(f"   - Embedding lookup is equivalent but much more efficient")

    print(f"\n❓ CONFUSION: 'Attention changes sequence length'")
    print(f"✅ REALITY: Attention preserves sequence length")
    print(f"   - Input to attention: (B, N, d_model)")
    print(
        f"   - Output from attention: (B, N, d_k) or (B, N, d_model) after projection"
    )
    print(f"   - Only the feature dimension might change, not sequence length")

    print(f"\n❓ CONFUSION: 'What are Q, K, V?'")
    print(f"✅ REALITY: Three different 'views' of the same input")
    print(f"   - All start from the same embeddings: (B, N, d_model)")
    print(f"   - Q (Query): 'What am I looking for?' -> (B, N, d_k)")
    print(f"   - K (Key): 'What do I contain?' -> (B, N, d_k)")
    print(f"   - V (Value): 'What information do I have?' -> (B, N, d_k)")
    print(f"   - Different linear projections of the same input")


if __name__ == "__main__":
    # Run the complete demonstration
    results = demonstrate_complete_pipeline()

    # Clarify common confusion points
    clarify_common_confusion()

    print(f"\n{'=' * 80}")
    print("SUMMARY: Your understanding is now complete! 🎉")
    print("=" * 80)
    print("✅ Token IDs -> Dense embeddings via lookup table")
    print("✅ Positional encodings added to give position information")
    print("✅ Attention operates on (B, N, d_model) embeddings")
    print("✅ D represents model dimension, NOT vocabulary size")
    print("✅ Attention preserves sequence length, transforms features")
