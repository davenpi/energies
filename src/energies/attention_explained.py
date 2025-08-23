"""
Attention Heads Explained: A Step-by-Step Breakdown

This module focuses specifically on understanding:
1. What Q, K, V actually represent
2. How attention scores are computed
3. Why we use multiple heads
4. The intuition behind the mechanism
"""

import math

import jax
import jax.numpy as jnp
from jax import random

# ============================================================================
# STEP-BY-STEP ATTENTION COMPUTATION
# ============================================================================


def explain_attention_step_by_step():
    """
    Walk through attention computation with concrete examples and intuition.
    """
    print("=" * 80)
    print("ATTENTION HEADS EXPLAINED: STEP BY STEP")
    print("=" * 80)

    # Setup a simple example
    key = random.PRNGKey(42)
    batch_size, seq_len, d_model = 1, 4, 8  # Small sizes for clarity

    # Create simple input embeddings
    # Think of these as: ["The", "cat", "sat", "down"]
    x = random.normal(key, (batch_size, seq_len, d_model))

    print(f"Input embeddings shape: {x.shape}")
    print(f"Think of this as 4 words, each represented by an 8-dimensional vector")
    print(f"Word embeddings:\n{x[0]}")  # Show first (and only) batch

    # ========================================================================
    # STEP 1: Create Q, K, V matrices
    # ========================================================================
    print(f"\n{'=' * 60}")
    print("STEP 1: WHAT ARE Q, K, V?")
    print("=" * 60)

    d_k = 4  # Head dimension (smaller than d_model for this example)

    # Initialize projection matrices
    key_q, key_k, key_v = random.split(key, 3)
    W_q = random.normal(key_q, (d_model, d_k)) * 0.1
    W_k = random.normal(key_k, (d_model, d_k)) * 0.1
    W_v = random.normal(key_v, (d_model, d_k)) * 0.1

    print(f"Projection matrices:")
    print(f"  W_q (Query projection): {W_q.shape} - 'What am I looking for?'")
    print(f"  W_k (Key projection):   {W_k.shape} - 'What do I contain?'")
    print(f"  W_v (Value projection): {W_v.shape} - 'What information do I have?'")

    # Compute Q, K, V
    Q = jnp.dot(x, W_q)  # (1, 4, 4)
    K = jnp.dot(x, W_k)  # (1, 4, 4)
    V = jnp.dot(x, W_v)  # (1, 4, 4)

    print(f"\nAfter projection:")
    print(f"  Q (Queries): {Q.shape}")
    print(f"  K (Keys):    {K.shape}")
    print(f"  V (Values):  {V.shape}")

    print(f"\n🧠 INTUITION:")
    print(f"  - Q[i] = 'What is word i looking for in other words?'")
    print(f"  - K[j] = 'What does word j offer/contain?'")
    print(f"  - V[j] = 'What information does word j have to share?'")

    # ========================================================================
    # STEP 2: Compute Attention Scores
    # ========================================================================
    print(f"\n{'=' * 60}")
    print("STEP 2: COMPUTING ATTENTION SCORES")
    print("=" * 60)

    # Raw attention scores: Q @ K^T
    raw_scores = jnp.matmul(Q, K.transpose(0, 2, 1))  # (1, 4, 4)

    print(f"Raw attention scores (Q @ K^T):")
    print(f"Shape: {raw_scores.shape}")
    print(f"Matrix (batch 0):\n{raw_scores[0]}")

    print(f"\n🔍 WHAT THIS MEANS:")
    print(f"  raw_scores[i,j] = how much word i's query matches word j's key")
    print(f"  Higher score = better match = more attention")

    # Scale by sqrt(d_k)
    scaled_scores = raw_scores / math.sqrt(d_k)

    print(f"\nAfter scaling by sqrt(d_k = {d_k}) = {math.sqrt(d_k):.2f}:")
    print(f"Scaled scores:\n{scaled_scores[0]}")

    print(f"\n❓ WHY SCALE?")
    print(f"  - Prevents scores from getting too large")
    print(f"  - Keeps gradients stable during training")
    print(f"  - Without scaling, softmax becomes too 'sharp'")

    # Apply softmax
    attention_weights = jax.nn.softmax(scaled_scores, axis=-1)

    print(f"\nAfter softmax (attention weights):")
    print(f"Attention weights:\n{attention_weights[0]}")

    print(f"\n✅ PROPERTIES OF ATTENTION WEIGHTS:")
    print(f"  - Each row sums to 1.0: {attention_weights[0].sum(axis=1)}")
    print(f"  - All values are between 0 and 1")
    print(f"  - Row i shows how much word i attends to each word")

    # ========================================================================
    # STEP 3: Apply Attention to Values
    # ========================================================================
    print(f"\n{'=' * 60}")
    print("STEP 3: APPLYING ATTENTION TO VALUES")
    print("=" * 60)

    # Weighted combination of values
    output = jnp.matmul(attention_weights, V)  # (1, 4, 4)

    print(f"Final output shape: {output.shape}")
    print(f"Output:\n{output[0]}")

    print(f"\n🎯 WHAT HAPPENED:")
    print(f"  - Each output[i] is a weighted combination of all V[j]")
    print(f"  - Weights come from how much word i attended to word j")
    print(f"  - High attention = more influence in the output")

    # Show the computation for one position
    pos = 1  # Second word
    print(f"\n📖 EXAMPLE: How output for position {pos} is computed:")
    print(f"  attention_weights[{pos}] = {attention_weights[0, pos]}")
    print(f"  output[{pos}] = ")
    for j in range(seq_len):
        weight = attention_weights[0, pos, j]
        print(f"    + {weight:.3f} * V[{j}] (word {j})")

    manual_output = jnp.sum(attention_weights[0, pos : pos + 1, :, None] * V[0], axis=1)
    print(f"  Manual calculation: {manual_output[0]}")
    print(f"  Automatic result:   {output[0, pos]}")
    print(f"  Match: {jnp.allclose(manual_output[0], output[0, pos])}")

    return Q, K, V, attention_weights, output


def demonstrate_multiple_heads():
    """
    Show why we use multiple attention heads.
    """
    print(f"\n{'=' * 80}")
    print("WHY MULTIPLE ATTENTION HEADS?")
    print("=" * 80)

    key = random.PRNGKey(123)
    batch_size, seq_len, d_model = 1, 6, 12
    n_heads = 3
    d_k = d_model // n_heads  # 4

    # Create input representing: ["The", "quick", "brown", "fox", "jumps", "high"]
    x = random.normal(key, (batch_size, seq_len, d_model))

    print(f"Input: 6 words, each with {d_model}-dim embeddings")
    print(f"Using {n_heads} heads, each with dimension {d_k}")

    # Create multiple heads with different random initializations
    heads = []
    keys = random.split(key, n_heads)

    for i, head_key in enumerate(keys):
        k_q, k_k, k_v = random.split(head_key, 3)
        head_params = {
            "W_q": random.normal(k_q, (d_model, d_k)) * 0.1,
            "W_k": random.normal(k_k, (d_model, d_k)) * 0.1,
            "W_v": random.normal(k_v, (d_model, d_k)) * 0.1,
        }
        heads.append(head_params)

    print(f"\n🎭 DIFFERENT HEADS LEARN DIFFERENT PATTERNS:")

    for i, head_params in enumerate(heads):
        # Compute attention for this head
        Q = jnp.dot(x, head_params["W_q"])
        K = jnp.dot(x, head_params["W_k"])
        V = jnp.dot(x, head_params["W_v"])

        scores = jnp.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(d_k)
        weights = jax.nn.softmax(scores, axis=-1)

        print(f"\nHead {i + 1} attention pattern:")
        print("     " + "".join([f"W{j:2d}" for j in range(seq_len)]))
        for pos in range(seq_len):
            row = f"W{pos:2d}: "
            for target in range(seq_len):
                weight = weights[0, pos, target]
                if weight > 0.3:
                    row += "██ "
                elif weight > 0.15:
                    row += "▓▓ "
                elif weight > 0.08:
                    row += "░░ "
                else:
                    row += "   "
            row += f" (max: {weights[0, pos].max():.2f})"
            print(row)

    print(f"\n🔍 WHAT DIFFERENT HEADS MIGHT LEARN:")
    print(f"  Head 1: Syntactic relationships (noun-verb, adj-noun)")
    print(f"  Head 2: Long-range dependencies (subject-verb agreement)")
    print(f"  Head 3: Local context (neighboring words)")
    print(f"  Head 4: Semantic similarity (related concepts)")

    print(f"\n💡 KEY INSIGHTS:")
    print(f"  - Each head has different W_q, W_k, W_v matrices")
    print(f"  - Different heads focus on different types of relationships")
    print(f"  - Multiple perspectives are combined for richer representation")
    print(f"  - Parallel computation: all heads run simultaneously")


def attention_intuition_examples():
    """
    Concrete examples of what attention learns to do.
    """
    print(f"\n{'=' * 80}")
    print("ATTENTION INTUITION: REAL-WORLD EXAMPLES")
    print("=" * 80)

    print(f"\n📚 EXAMPLE 1: 'The cat that I saw yesterday was sleeping'")
    print(f"  Word positions: [The, cat, that, I, saw, yesterday, was, sleeping]")
    print(f"  Question: What was sleeping?")
    print(f"  ")
    print(f"  Attention pattern for 'sleeping':")
    print(f"    sleeping -> The:       0.05  (low)")
    print(f"    sleeping -> cat:       0.70  (HIGH - this is the subject!)")
    print(f"    sleeping -> that:      0.02  (low)")
    print(f"    sleeping -> I:         0.03  (low)")
    print(f"    sleeping -> saw:       0.05  (low)")
    print(f"    sleeping -> yesterday: 0.05  (low)")
    print(f"    sleeping -> was:       0.10  (medium - auxiliary verb)")
    print(f"  ")
    print(f"  🎯 The model learns: 'sleeping' should pay attention to 'cat'")

    print(f"\n📚 EXAMPLE 2: 'She gave him the book'")
    print(f"  Attention pattern for 'gave':")
    print(f"    gave -> She:  0.40  (subject - who is giving)")
    print(f"    gave -> him:  0.35  (indirect object - to whom)")
    print(f"    gave -> book: 0.20  (direct object - what is given)")
    print(f"  ")
    print(f"  🎯 The model learns: verbs attend to their arguments")

    print(f"\n📚 EXAMPLE 3: Translation - 'Le chat noir' -> 'The black cat'")
    print(f"  French: [Le, chat, noir]")
    print(f"  English: [The, black, cat]")
    print(f"  ")
    print(f"  Cross-attention pattern:")
    print(f"    'black' -> Le:   0.05")
    print(f"    'black' -> chat: 0.10")
    print(f"    'black' -> noir: 0.85  (HIGH - 'black' comes from 'noir')")
    print(f"  ")
    print(f"  🎯 The model learns: align words across languages")

    print(f"\n🧠 WHAT MAKES ATTENTION POWERFUL:")
    print(f"  1. DYNAMIC: Attention changes based on context")
    print(f"  2. SELECTIVE: Focus on relevant information, ignore noise")
    print(f"  3. DIFFERENTIABLE: Can be trained end-to-end")
    print(f"  4. PARALLELIZABLE: All positions computed simultaneously")
    print(f"  5. INTERPRETABLE: Can visualize what the model focuses on")


def common_attention_misconceptions():
    """
    Clear up common misunderstandings about attention.
    """
    print(f"\n{'=' * 80}")
    print("COMMON ATTENTION MISCONCEPTIONS")
    print("=" * 80)

    print(f"\n❌ MISCONCEPTION 1: 'Q, K, V are different inputs'")
    print(f"✅ REALITY: Q, K, V are different VIEWS of the same input")
    print(f"   - All three come from the same embeddings X")
    print(f"   - Q = X @ W_q,  K = X @ W_k,  V = X @ W_v")
    print(f"   - Different projection matrices create different perspectives")

    print(f"\n❌ MISCONCEPTION 2: 'Attention weights are learned parameters'")
    print(f"✅ REALITY: Attention weights are COMPUTED dynamically")
    print(f"   - W_q, W_k, W_v are the learned parameters")
    print(f"   - Attention weights change for every input sequence")
    print(f"   - They're computed as softmax(Q @ K^T / sqrt(d_k))")

    print(f"\n❌ MISCONCEPTION 3: 'Each head looks at different positions'")
    print(f"✅ REALITY: Each head can look at ALL positions")
    print(f"   - All heads receive the full sequence")
    print(f"   - Heads learn to focus on different TYPES of relationships")
    print(f"   - Some might focus locally, others on long-range dependencies")

    print(f"\n❌ MISCONCEPTION 4: 'Attention replaces RNNs completely'")
    print(f"✅ REALITY: Attention has different trade-offs")
    print(f"   - Attention: Parallel, long-range, but O(n²) memory")
    print(f"   - RNNs: Sequential, O(n) memory, but harder to parallelize")
    print(f"   - Choice depends on your specific use case")

    print(f"\n❌ MISCONCEPTION 5: 'More heads = always better'")
    print(f"✅ REALITY: There's a sweet spot")
    print(f"   - Too few heads: Limited expressiveness")
    print(f"   - Too many heads: Redundancy, harder to train")
    print(f"   - Common choices: 8, 12, 16 heads for large models")


if __name__ == "__main__":
    # Step-by-step explanation
    Q, K, V, weights, output = explain_attention_step_by_step()

    # Multiple heads demonstration
    demonstrate_multiple_heads()

    # Real-world intuition
    attention_intuition_examples()

    # Clear up misconceptions
    common_attention_misconceptions()

    print(f"\n{'=' * 80}")
    print("🎉 ATTENTION HEADS: FULLY EXPLAINED!")
    print("=" * 80)
    print("✅ Q, K, V are different projections of the same input")
    print("✅ Attention scores = similarity between queries and keys")
    print("✅ Softmax ensures attention weights sum to 1")
    print("✅ Output = weighted combination of values")
    print("✅ Multiple heads capture different relationship types")
    print("✅ Everything is differentiable and trainable")
    print("\n🚀 Ready to build complete transformer blocks!")
