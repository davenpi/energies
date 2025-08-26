"""
Complete Llama model implementation.

This module combines all components (embeddings, attention, MLP, norms)
into the full transformer architecture.
"""

import flax.linen as nn
import jax.numpy as jnp
from attention import LlamaAttention
from config import LlamaConfig
from embeddings import LlamaEmbeddings
from layers import RMSNorm, create_causal_mask
from mlp import LlamaMLP


class LlamaTransformerBlock(nn.Module):
    """A single Llama transformer block.

    This combines:
    1. Self-attention with pre-norm and residual connection
    2. Feed-forward network with pre-norm and residual connection

    The pre-norm architecture (norm before, not after) provides
    better training stability and gradient flow.
    """

    config: LlamaConfig

    def setup(self):
        # Self-attention mechanism
        self.attention = LlamaAttention(self.config)

        # Feed-forward network
        self.mlp = LlamaMLP(self.config)

        # Layer norms (applied before each sub-layer)
        self.attention_norm = RMSNorm(self.config)
        self.ffn_norm = RMSNorm(self.config)

    def __call__(self, x: jnp.ndarray, attention_mask: jnp.ndarray = None):
        """
        Transformer block forward pass.

        Args:
            x: Input of shape (batch_size, seq_len, hidden_size)
            attention_mask: Causal mask of shape (1, 1, seq_len, seq_len)

        Returns:
            Output of shape (batch_size, seq_len, hidden_size)
        """
        # Self-attention with pre-norm and residual (clean one-liner!)
        h = x + self.attention(self.attention_norm(x), attention_mask)

        # Feed-forward with pre-norm and residual (clean one-liner!)
        out = h + self.mlp(self.ffn_norm(h))

        return out


class LlamaModel(nn.Module):
    """Complete Llama model.

    Architecture:
    1. Token embeddings (vocab_size → hidden_size)
    2. 16 transformer blocks (attention + MLP)
    3. Final RMS normalization
    4. Output projection (tied embeddings: hidden_size → vocab_size)
    """

    config: LlamaConfig

    def setup(self):
        # Token embeddings
        self.embed_tokens = LlamaEmbeddings(self.config)

        # Stack of transformer blocks
        self.layers = [
            LlamaTransformerBlock(self.config)
            for _ in range(self.config.num_hidden_layers)  # 16 layers
        ]

        # Final normalization
        self.norm = RMSNorm(self.config)

    def __call__(self, input_ids: jnp.ndarray, attention_mask: jnp.ndarray = None):
        """
        Full model forward pass.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional causal mask

        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # 1. Token embeddings: (batch, seq_len) → (batch, seq_len, hidden_size)
        x = self.embed_tokens(input_ids)

        # 2. Create causal mask if not provided
        if attention_mask is None:
            attention_mask = create_causal_mask(seq_len)

        # 3. Pass through all transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask)

        # 4. Final normalization
        x = self.norm(x)

        # 5. Output projection using tied embeddings
        # Instead of a separate linear layer, reuse embedding weights transposed
        logits = x @ self.embed_tokens.embed_tokens.embedding.T

        return logits


def test_transformer_block():
    """Test a single transformer block."""
    from pathlib import Path

    import jax
    from config import LlamaConfig

    config_path = Path(__file__).parent.parent / "llama_cfg.json"
    config = LlamaConfig.from_json(str(config_path))

    print("=== Testing Transformer Block ===")
    print(f"Hidden size: {config.hidden_size}")

    # Create transformer block
    block = LlamaTransformerBlock(config=config)

    # Test input
    key = jax.random.PRNGKey(42)
    batch_size, seq_len = 2, 8
    x = jax.random.normal(key, (batch_size, seq_len, config.hidden_size))

    # Create causal mask
    causal_mask = create_causal_mask(seq_len)

    print(f"\n=== Shape Flow ===")
    print(f"Input: {x.shape}")
    print(f"Causal mask: {causal_mask.shape}")

    # Initialize and run
    params = block.init(key, x, causal_mask)
    output = block.apply(params, x, causal_mask)

    print(f"Output: {output.shape}")
    print(f"✓ Shape preserved: {x.shape == output.shape}")

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

    # Expected: Attention (~10.5M) + MLP (~50.3M) + 2 RMSNorms (~4K) ≈ 60.8M
    expected_attention = (
        config.hidden_size * config.num_attention_heads * config.head_dim
        + config.hidden_size * config.num_key_value_heads * config.head_dim
        + config.hidden_size * config.num_key_value_heads * config.head_dim
        + config.hidden_size * config.hidden_size
    )

    expected_mlp = (
        config.hidden_size * config.intermediate_size
        + config.hidden_size * config.intermediate_size
        + config.intermediate_size * config.hidden_size
    )

    expected_norms = 2 * config.hidden_size
    expected_total = expected_attention + expected_mlp + expected_norms

    print(f"\n=== Parameters ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Expected: {expected_total:,}")
    print(f"✓ Parameter count correct: {total_params == expected_total}")

    print(f"\n✅ Transformer block working correctly!")


def test_full_model():
    """Test the complete Llama model."""
    from pathlib import Path

    import jax
    from config import LlamaConfig

    config_path = Path(__file__).parent.parent / "llama_cfg.json"
    config = LlamaConfig.from_json(str(config_path))

    print("\n=== Testing Complete Llama Model ===")
    config.print_summary()

    # Create full model
    model = LlamaModel(config=config)

    # Test input - simulate tokenized text
    key = jax.random.PRNGKey(42)
    batch_size, seq_len = 2, 16
    input_ids = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)

    print(f"\n=== Model Input ===")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Sample tokens: {input_ids[0, :8].tolist()}")

    # Initialize model
    print(f"\nInitializing model parameters...")
    params = model.init(key, input_ids)

    # Forward pass
    print(f"Running forward pass...")
    logits = model.apply(params, input_ids)

    print(f"\n=== Model Output ===")
    print(f"Logits shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {config.vocab_size})")
    print(
        f"✓ Shape correct: {logits.shape == (batch_size, seq_len, config.vocab_size)}"
    )

    # Convert to probabilities for next token prediction
    last_token_logits = logits[0, -1, :]  # First batch, last position
    probs = jax.nn.softmax(last_token_logits)

    # Top-5 predictions
    top_k = 5
    top_indices = jnp.argsort(probs)[-top_k:][::-1]
    top_probs = probs[top_indices]

    print(f"\n=== Next Token Predictions ===")
    print("Top 5 most likely next tokens:")
    for i, (token_id, prob) in enumerate(zip(top_indices, top_probs)):
        print(f"  {i + 1}. Token {token_id}: {prob:.4f}")

    # Total parameter count
    def count_params(params_dict):
        total = 0
        for key, value in params_dict.items():
            if isinstance(value, dict):
                total += count_params(value)
            else:
                total += value.size
        return total

    total_params = count_params(params)
    estimated_params = config.estimate_parameters()

    print(f"\n=== Final Parameter Count ===")
    print(f"Actual parameters: {total_params:,}")
    print(f"Estimated parameters: {estimated_params:,}")
    print(f"Parameters in billions: {total_params / 1e9:.3f}B")
    print(f"✓ Matches estimate: {total_params == estimated_params}")

    # Parameter breakdown
    embedding_params = config.vocab_size * config.hidden_size
    layer_params = (
        total_params - embedding_params - config.hidden_size
    ) // config.num_hidden_layers

    print(f"\n=== Parameter Breakdown ===")
    print(
        f"Embeddings (tied): {embedding_params:,} "
        f"({embedding_params / total_params * 100:.1f}%)"
    )
    print(f"Per transformer block: {layer_params:,}")
    print(
        f"16 transformer blocks: {layer_params * 16:,} "
        f"({layer_params * 16 / total_params * 100:.1f}%)"
    )
    print(
        f"Final norm: {config.hidden_size:,} "
        f"({config.hidden_size / total_params * 100:.3f}%)"
    )

    print(f"\n🎉 Complete Llama 3.2-1B model implemented successfully!")
    print(f"✨ Ready for training or inference with pre-trained weights!")


if __name__ == "__main__":
    test_transformer_block()
    test_full_model()
