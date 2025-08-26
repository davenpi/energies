"""
Llama model configuration.
"""

import json
from pathlib import Path

from flax import struct


@struct.dataclass
class LlamaConfig:
    """Llama model configuration.

    This dataclass holds all the architectural parameters that define
    the model structure. Understanding these dimensions is key to
    understanding how shapes flow through the model.
    """

    # === Core Architecture ===
    vocab_size: int = 128256  # Size of tokenizer vocabulary
    hidden_size: int = 2048  # Main model dimension (d_model)
    intermediate_size: int = 8192  # FFN hidden dimension (4x hidden_size)
    num_hidden_layers: int = 16  # Number of transformer blocks

    # === Attention Configuration ===
    num_attention_heads: int = 32  # Number of query heads
    num_key_value_heads: int = 8  # Number of key/value heads (GQA)
    head_dim: int = 64  # Dimension per attention head

    # === Position Encoding ===
    max_position_embeddings: int = 131072  # Maximum sequence length (128k)
    rope_theta: float = 500000.0  # RoPE base frequency

    # === Normalization ===
    rms_norm_eps: float = 1e-5  # RMSNorm epsilon

    # === Model Behavior ===
    tie_word_embeddings: bool = True  # Share input/output embeddings

    # === RoPE Scaling ===
    rope_theta: float = 500000.0
    rope_scaling_factor: float = 32.0
    rope_high_freq_factor: float = 4.0
    rope_low_freq_factor: float = 1.0
    original_max_position_embeddings: int = 8192
    rope_type: str = "llama3"

    @classmethod
    def from_json(cls, config_path: str) -> "LlamaConfig":
        """Load configuration from JSON file.

        Args:
            config_path: Path to the JSON configuration file

        Returns:
            LlamaConfig instance with loaded parameters
        """
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Handle rope_scaling nested dict
        rope_scaling = config_dict.get("rope_scaling", {})

        return cls(
            vocab_size=config_dict["vocab_size"],
            hidden_size=config_dict["hidden_size"],
            intermediate_size=config_dict["intermediate_size"],
            num_hidden_layers=config_dict["num_hidden_layers"],
            num_attention_heads=config_dict["num_attention_heads"],
            num_key_value_heads=config_dict["num_key_value_heads"],
            head_dim=config_dict["head_dim"],
            max_position_embeddings=config_dict["max_position_embeddings"],
            rope_theta=config_dict["rope_theta"],
            rope_scaling_factor=rope_scaling.get("factor", 1.0),
            rope_high_freq_factor=rope_scaling.get("high_freq_factor", 1.0),
            rope_low_freq_factor=rope_scaling.get("low_freq_factor", 1.0),
            original_max_position_embeddings=rope_scaling.get(
                "original_max_position_embeddings", 8192
            ),
            rope_type=rope_scaling.get("rope_type", "default"),
            rms_norm_eps=config_dict["rms_norm_eps"],
            tie_word_embeddings=config_dict["tie_word_embeddings"],
        )

    def print_summary(self):
        """Print a human-readable summary of the configuration."""
        print("=== Llama Model Configuration ===")
        print(f"Model size: ~{self.estimate_parameters() / 1e9:.1f}B parameters")
        print(f"Vocabulary: {self.vocab_size:,} tokens")
        print(f"Max sequence: {self.max_position_embeddings:,} tokens")
        print()
        print("Architecture:")
        print(f"  Layers: {self.num_hidden_layers}")
        print(f"  Hidden size: {self.hidden_size}")
        print(
            f"  FFN size: {self.intermediate_size} "
            f"({self.intermediate_size / self.hidden_size:.1f}x)"
        )
        print()
        print("Attention:")
        print(f"  Query heads: {self.num_attention_heads}")
        print(f"  Key/Value heads: {self.num_key_value_heads}")
        print(f"  Head dimension: {self.head_dim}")
        print(f"  GQA ratio: {self.num_attention_heads // self.num_key_value_heads}:1")
        print()
        print("Key shape flows:")
        print(f"  Input: (batch, seq_len, {self.hidden_size})")
        print(f"  Q: (batch, seq_len, {self.num_attention_heads}, {self.head_dim})")
        print(f"  K,V: (batch, seq_len, {self.num_key_value_heads}, {self.head_dim})")
        print(f"  FFN intermediate: (batch, seq_len, {self.intermediate_size})")

    def estimate_parameters(self) -> int:
        """Estimate total model parameters."""
        # Embeddings (tied, so only count once)
        embedding_params = self.vocab_size * self.hidden_size

        # Per-layer parameters
        attention_params = (
            self.hidden_size * self.num_attention_heads * self.head_dim  # q_proj
            + self.hidden_size * self.num_key_value_heads * self.head_dim  # k_proj
            + self.hidden_size * self.num_key_value_heads * self.head_dim  # v_proj
            + self.hidden_size * self.hidden_size  # o_proj
        )

        mlp_params = (
            self.hidden_size * self.intermediate_size  # gate_proj
            + self.hidden_size * self.intermediate_size  # up_proj
            + self.intermediate_size * self.hidden_size  # down_proj
        )

        norm_params = 2 * self.hidden_size  # 2 RMSNorms per layer

        layer_params = attention_params + mlp_params + norm_params
        total_layer_params = self.num_hidden_layers * layer_params

        # Final norm
        final_norm_params = self.hidden_size

        return embedding_params + total_layer_params + final_norm_params


# Test the config
def test_config():
    """Test configuration loading and parameter estimation."""
    config_path = Path(__file__).parent.parent / "llama_cfg.json"
    config = LlamaConfig.from_json(str(config_path))

    config.print_summary()

    print("\n=== Validation ===")
    print(f"✓ Config loaded successfully")
    print(f"✓ Parameter estimate: {config.estimate_parameters():,}")


if __name__ == "__main__":
    test_config()
