"""
Causal Attention: Understanding Masking in Language Models

This module demonstrates:
1. What causal masking is and why we need it
2. How to implement causal attention
3. The difference between causal and bidirectional attention
4. When to use each type
"""

import math

import jax
import jax.numpy as jnp
from jax import random


def create_causal_mask(seq_len):
    """
    Create a causal mask that prevents attention to future positions.

    Args:
        seq_len: Length of the sequence

    Returns:
        mask: Boolean mask of shape (seq_len, seq_len)
              True = allowed, False = masked (set to -inf)
    """
    # Create lower triangular matrix (including diagonal)
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    return mask.astype(bool)


def apply_causal_mask(attention_scores, mask):
    """
    Apply causal mask to attention scores.

    Args:
        attention_scores: Raw attention scores (batch_size, seq_len, seq_len)
        mask: Causal mask (seq_len, seq_len)

    Returns:
        masked_scores: Attention scores with future positions set to -inf
    """
    # Where mask is False, set to large negative number
    large_negative = -1e9
    masked_scores = jnp.where(mask, attention_scores, large_negative)
    return masked_scores


def causal_attention_forward(params, x, use_causal_mask=True):
    """
    Attention forward pass with optional causal masking.

    Args:
        params: Dictionary with W_q, W_k, W_v matrices
        x: Input tensor (batch_size, seq_len, d_model)
        use_causal_mask: Whether to apply causal masking

    Returns:
        output: Attention output
        attention_weights: Attention weights (masked if causal)
    """
    # Standard Q, K, V computation
    Q = jnp.dot(x, params["W_q"])
    K = jnp.dot(x, params["W_k"])
    V = jnp.dot(x, params["W_v"])

    d_k = Q.shape[-1]
    seq_len = Q.shape[1]

    # Compute raw attention scores
    scores = jnp.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(d_k)

    # Apply causal mask if requested
    if use_causal_mask:
        mask = create_causal_mask(seq_len)
        scores = apply_causal_mask(scores, mask)

    # Apply softmax
    attention_weights = jax.nn.softmax(scores, axis=-1)

    # Compute output
    output = jnp.matmul(attention_weights, V)

    return output, attention_weights


def demonstrate_causal_vs_bidirectional():
    """
    Show the difference between causal and bidirectional attention.
    """
    print("=" * 80)
    print("CAUSAL vs BIDIRECTIONAL ATTENTION")
    print("=" * 80)

    # Setup
    key = random.PRNGKey(42)
    batch_size, seq_len, d_model = 1, 6, 8
    d_k = 4

    # Create input representing: ["I", "love", "deep", "learning", "very", "much"]
    x = random.normal(key, (batch_size, seq_len, d_model))

    # Initialize attention parameters
    keys = random.split(key, 3)
    params = {
        "W_q": random.normal(keys[0], (d_model, d_k)) * 0.1,
        "W_k": random.normal(keys[1], (d_model, d_k)) * 0.1,
        "W_v": random.normal(keys[2], (d_model, d_k)) * 0.1,
    }

    words = ["I", "love", "deep", "learning", "very", "much"]
    print(f"Input sequence: {' '.join(words)}")
    print(f"Positions:      {' '.join([f'{i:4d}' for i in range(len(words))])}")

    # ========================================================================
    # Bidirectional Attention (BERT-style)
    # ========================================================================
    print(f"\n{'=' * 50}")
    print("BIDIRECTIONAL ATTENTION (BERT-style)")
    print("=" * 50)

    _, bidirectional_weights = causal_attention_forward(
        params, x, use_causal_mask=False
    )

    print("Attention pattern (each word can attend to ALL words):")
    print("From\\To  " + "".join([f"{word:>8}" for word in words]))

    for i, from_word in enumerate(words):
        row = f"{from_word:>7}: "
        for j in range(seq_len):
            weight = bidirectional_weights[0, i, j]
            row += f"{weight:8.3f}"
        print(row)

    print(f"\n🔍 NOTICE:")
    print(f"  - Each word can attend to words BEFORE and AFTER it")
    print(f"  - 'deep' can attend to 'learning' (future context)")
    print(f"  - Good for: classification, question answering, fill-in-the-blank")

    # ========================================================================
    # Causal Attention (GPT-style)
    # ========================================================================
    print(f"\n{'=' * 50}")
    print("CAUSAL ATTENTION (GPT-style)")
    print("=" * 50)

    _, causal_weights = causal_attention_forward(params, x, use_causal_mask=True)

    print("Attention pattern (each word can only attend to PREVIOUS words):")
    print("From\\To  " + "".join([f"{word:>8}" for word in words]))

    for i, from_word in enumerate(words):
        row = f"{from_word:>7}: "
        for j in range(seq_len):
            weight = causal_weights[0, i, j]
            if j > i:  # Future positions should be 0
                row += f"{'---':>8}"
            else:
                row += f"{weight:8.3f}"
        print(row)

    print(f"\n🔍 NOTICE:")
    print(f"  - Each word can ONLY attend to itself and previous words")
    print(f"  - 'deep' CANNOT attend to 'learning' (future is masked)")
    print(f"  - Future positions show '---' (actually 0 after softmax)")
    print(f"  - Good for: text generation, next token prediction")

    # ========================================================================
    # Show the mask
    # ========================================================================
    print(f"\n{'=' * 50}")
    print("THE CAUSAL MASK")
    print("=" * 50)

    mask = create_causal_mask(seq_len)
    print("Causal mask (True = allowed, False = blocked):")
    print("From\\To  " + "".join([f"{i:>5}" for i in range(seq_len)]))

    for i in range(seq_len):
        row = f"{i:>5}:   "
        for j in range(seq_len):
            if mask[i, j]:
                row += "  ✓  "
            else:
                row += "  ✗  "
        print(row)

    print(f"\n💡 HOW THE MASK WORKS:")
    print(f"  1. Compute raw attention scores: Q @ K^T")
    print(f"  2. Where mask is False, set scores to -∞")
    print(f"  3. Apply softmax: exp(-∞) = 0")
    print(f"  4. Result: zero attention to future positions")


def language_generation_example():
    """
    Show why causality is essential for language generation.
    """
    print(f"\n{'=' * 80}")
    print("WHY CAUSALITY IS ESSENTIAL FOR LANGUAGE GENERATION")
    print("=" * 80)

    print(f"\n📝 SCENARIO: Generating text word by word")
    print(f"   Prompt: 'The weather today is'")
    print(f"   Goal: Generate next word")

    print(f"\n🤖 WITHOUT CAUSAL MASKING (cheating):")
    print(f"   Input:  ['The', 'weather', 'today', 'is', 'sunny']")
    print(f"   Model can see 'sunny' when predicting what comes after 'is'")
    print(f"   This is CHEATING - the model has access to the answer!")

    print(f"\n✅ WITH CAUSAL MASKING (correct):")
    print(f"   Step 1: ['The'] -> predict 'weather'")
    print(f"   Step 2: ['The', 'weather'] -> predict 'today'")
    print(f"   Step 3: ['The', 'weather', 'today'] -> predict 'is'")
    print(f"   Step 4: ['The', 'weather', 'today', 'is'] -> predict 'sunny'")
    print(f"   Each step only sees previous context (no cheating!)")

    print(f"\n🎯 TRAINING vs INFERENCE:")
    print(f"   TRAINING: Process full sequence with causal mask")
    print(f"     - Input:  ['The', 'weather', 'today', 'is', 'sunny']")
    print(f"     - Target: ['weather', 'today', 'is', 'sunny', '<END>']")
    print(f"     - Mask prevents seeing future tokens")
    print(f"   ")
    print(f"   INFERENCE: Generate one token at a time")
    print(f"     - Start: ['The'] -> generate 'weather'")
    print(f"     - Next:  ['The', 'weather'] -> generate 'today'")
    print(f"     - Continue until <END> token")


def when_to_use_each_type():
    """
    Guide on when to use causal vs bidirectional attention.
    """
    print(f"\n{'=' * 80}")
    print("WHEN TO USE CAUSAL vs BIDIRECTIONAL ATTENTION")
    print("=" * 80)

    print(f"\n🔒 USE CAUSAL ATTENTION FOR:")
    print(f"   ✅ Text Generation (GPT, ChatGPT)")
    print(f"      - Next token prediction")
    print(f"      - Story/code completion")
    print(f"      - Dialogue systems")
    print(f"   ")
    print(f"   ✅ Autoregressive Tasks")
    print(f"      - Language modeling")
    print(f"      - Time series forecasting")
    print(f"      - Sequential decision making")
    print(f"   ")
    print(f"   ✅ Decoder in Seq2Seq Models")
    print(f"      - Translation (target side)")
    print(f"      - Summarization (output side)")
    print(f"      - Any generation task")

    print(f"\n🔓 USE BIDIRECTIONAL ATTENTION FOR:")
    print(f"   ✅ Understanding Tasks (BERT)")
    print(f"      - Text classification")
    print(f"      - Sentiment analysis")
    print(f"      - Named entity recognition")
    print(f"   ")
    print(f"   ✅ Fill-in-the-Blank Tasks")
    print(f"      - Masked language modeling")
    print(f"      - Cloze tests")
    print(f"      - Missing word prediction")
    print(f"   ")
    print(f"   ✅ Encoder in Seq2Seq Models")
    print(f"      - Translation (source side)")
    print(f"      - Summarization (input side)")
    print(f"      - Question answering")

    print(f"\n🏗️ ARCHITECTURE PATTERNS:")
    print(f"   📖 BERT: Bidirectional encoder only")
    print(f"   🤖 GPT: Causal decoder only")
    print(f"   🔄 T5: Bidirectional encoder + Causal decoder")
    print(f"   🌍 Transformer: Bidirectional encoder + Causal decoder")


def implementation_tips():
    """
    Practical tips for implementing causal attention.
    """
    print(f"\n{'=' * 80}")
    print("IMPLEMENTATION TIPS")
    print("=" * 80)

    print(f"\n💻 EFFICIENT MASKING:")
    print(f"   # Create mask once, reuse for all batches")
    print(f"   mask = jnp.tril(jnp.ones((seq_len, seq_len)))")
    print(f"   ")
    print(f"   # Apply before softmax (not after)")
    print(f"   scores = jnp.where(mask, scores, -1e9)")
    print(f"   weights = jax.nn.softmax(scores, axis=-1)")

    print(f"\n⚡ PERFORMANCE CONSIDERATIONS:")
    print(f"   - Mask is O(seq_len²) memory")
    print(f"   - For very long sequences, consider:")
    print(f"     • Sliding window attention")
    print(f"     • Sparse attention patterns")
    print(f"     • Linear attention variants")

    print(f"\n🐛 COMMON BUGS:")
    print(f"   ❌ Applying mask after softmax (too late!)")
    print(f"   ❌ Using 0 instead of -∞ for masking")
    print(f"   ❌ Wrong mask shape (should be seq_len × seq_len)")
    print(f"   ❌ Forgetting to mask during training")


if __name__ == "__main__":
    # Demonstrate causal vs bidirectional
    demonstrate_causal_vs_bidirectional()

    # Show why causality matters for generation
    language_generation_example()

    # Guide on when to use each
    when_to_use_each_type()

    # Implementation tips
    implementation_tips()

    print(f"\n{'=' * 80}")
    print("🎉 CAUSAL ATTENTION: FULLY EXPLAINED!")
    print("=" * 80)
    print("✅ Causal = only attend to current and previous positions")
    print("✅ Essential for text generation and autoregressive tasks")
    print("✅ Implemented with triangular mask before softmax")
    print("✅ Bidirectional attention for understanding tasks")
    print("✅ Choose based on your specific use case")
    print("\n🚀 Ready to build complete transformer architectures!")
