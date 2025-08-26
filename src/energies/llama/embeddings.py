"""
Token embeddings for Llama.

This is the entry point of the model - converts discrete token IDs
into continuous vector representations that the transformer can work with.
"""

import flax.linen as nn
import jax.numpy as jnp
from config import LlamaConfig


class LlamaEmbeddings(nn.Module):
    """Token embeddings layer.

    Transforms token IDs into dense vector representations.
    This is where discrete language becomes continuous math.
    """

    config: LlamaConfig

    def setup(self):
        self.embed_tokens = nn.Embed(
            num_embeddings=self.config.vocab_size,  # 128,256 tokens
            features=self.config.hidden_size,  # 2048 dimensions
            embedding_init=nn.initializers.normal(stddev=0.02),
        )

    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        """
        Convert token IDs to embeddings.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)

        Returns:
            embeddings: Dense vectors of shape (batch_size, seq_len, hidden_size)
        """
        return self.embed_tokens(input_ids)


def test_embeddings():
    """Test embeddings with clear shape tracing."""
    from pathlib import Path

    import jax
    from config import LlamaConfig

    # Load config
    config_path = Path(__file__).parent.parent / "llama_cfg.json"
    config = LlamaConfig.from_json(str(config_path))

    print("=== Testing Llama Embeddings ===")
    print(f"Vocabulary size: {config.vocab_size:,}")
    print(f"Hidden size: {config.hidden_size}")

    # Create embeddings
    embeddings = LlamaEmbeddings(config=config)

    # Test input - simulate tokenized text
    key = jax.random.PRNGKey(42)
    batch_size, seq_len = 2, 10

    # Random token IDs (in practice, these come from a tokenizer)
    input_ids = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)

    print(f"\n=== Shape Transformation ===")
    print(f"Input token IDs: {input_ids.shape}")
    print(f"Sample tokens: {input_ids[0, :5].tolist()}")

    # Initialize and run
    params = embeddings.init(key, input_ids)
    output = embeddings.apply(params, input_ids)

    print(f"Output embeddings: {output.shape}")
    print(
        f"✓ Shape correct: (batch={batch_size}, seq={seq_len},"
        f" hidden_size={config.hidden_size})"
    )

    # Show parameter count
    embedding_matrix = params["params"]["embed_tokens"]["embedding"]
    param_count = embedding_matrix.size
    expected = config.vocab_size * config.hidden_size

    print(f"\n=== Parameters ===")
    print(f"Embedding matrix: {embedding_matrix.shape}")
    print(f"Parameter count: {param_count:,}")
    print(f"Expected: {expected:,}")
    print(f"✓ Parameters correct: {param_count == expected}")

    # Show what embeddings look like
    print(f"\n=== Sample Embedding ===")
    first_token_embedding = output[0, 0, :5]  # First 5 dims of first token
    print(
        f"Token {input_ids[0, 0]} → [{first_token_embedding[0]:.3f},"
        f"{first_token_embedding[1]:.3f}, {first_token_embedding[2]:.3f}, ..."
        f"]"
    )

    print(f"\n✅ Embeddings working correctly!")


if __name__ == "__main__":
    test_embeddings()
