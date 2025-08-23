import time

import jax
import jax.numpy as jnp
from jax import random

# ============================================================================
# JAX MLP Implementation from Scratch
# ============================================================================


def init_layer_params(key, n_in, n_out):
    """Initialize parameters for a single layer using Xavier initialization."""
    w_key, b_key = random.split(key)
    # Xavier initialization: scale by sqrt(1/n_in)
    scale = jnp.sqrt(1.0 / n_in)
    W = random.normal(w_key, (n_in, n_out)) * scale
    b = jnp.zeros(n_out)
    return {"W": W, "b": b}


def init_mlp_params(key, layer_sizes):
    """Initialize parameters for the entire MLP."""
    keys = random.split(key, len(layer_sizes) - 1)
    params = []
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_params = init_layer_params(keys[i], n_in, n_out)
        params.append(layer_params)
    return params


def relu(x):
    """ReLU activation function."""
    return jnp.maximum(0, x)


def forward_layer(params, x):
    """Forward pass through a single layer."""
    return jnp.dot(x, params["W"]) + params["b"]


def mlp_forward(params, x):
    """Forward pass through the entire MLP."""
    # Forward through all hidden layers with ReLU
    for layer_params in params[:-1]:
        x = forward_layer(layer_params, x)
        x = relu(x)

    # Final layer (no activation for regression, or you could add softmax for classification)
    x = forward_layer(params[-1], x)
    return x


def mse_loss(params, x, y):
    """Mean squared error loss."""
    predictions = mlp_forward(params, x)
    return jnp.mean((predictions - y) ** 2)


# Compile the loss function and get its gradient
loss_and_grad = jax.jit(jax.value_and_grad(mse_loss))


def update_params(params, grads, learning_rate):
    """Update parameters using gradient descent."""
    updated_params = []
    for layer_params, layer_grads in zip(params, grads):
        updated_layer = {}
        for key in layer_params:
            updated_layer[key] = layer_params[key] - learning_rate * layer_grads[key]
        updated_params.append(updated_layer)
    return updated_params


# ============================================================================
# Example: Training on a simple regression task
# ============================================================================


def generate_data(key, n_samples=1000, n_features=10):
    """Generate synthetic regression data."""
    X_key, y_key, noise_key = random.split(key, 3)

    # Generate random input features
    X = random.normal(X_key, (n_samples, n_features))

    # Create a simple target: y = sum of first 3 features + noise
    true_weights = jnp.array([1.5, -2.0, 0.8] + [0.0] * (n_features - 3))
    y = jnp.dot(X, true_weights) + 0.1 * random.normal(noise_key, (n_samples,))

    return X, y.reshape(-1, 1)  # Reshape y to be 2D


def train_mlp():
    """Train the MLP on synthetic data."""
    # Set random seed for reproducibility
    key = random.PRNGKey(42)
    data_key, params_key = random.split(key)

    # Generate data
    X_train, y_train = generate_data(data_key, n_samples=1000, n_features=10)

    # Define network architecture
    layer_sizes = [10, 64, 32, 1]  # input_dim, hidden1, hidden2, output_dim

    # Initialize parameters
    params = init_mlp_params(params_key, layer_sizes)

    # Training hyperparameters
    learning_rate = 0.01
    n_epochs = 1000

    print("Training MLP...")
    print(f"Architecture: {layer_sizes}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {n_epochs}")
    print("-" * 50)

    # Training loop
    for epoch in range(n_epochs):
        # Compute loss and gradients
        loss_val, grads = loss_and_grad(params, X_train, y_train)

        # Update parameters
        params = update_params(params, grads, learning_rate)

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}, Loss: {loss_val:.6f}")

    # Final evaluation
    final_loss = mse_loss(params, X_train, y_train)
    predictions = mlp_forward(params, X_train)

    print("-" * 50)
    print(f"Final training loss: {final_loss:.6f}")
    print(f"Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"Target range: [{y_train.min():.3f}, {y_train.max():.3f}]")

    return params, X_train, y_train


# ============================================================================
# Advanced: Using JAX transformations
# ============================================================================


def demonstrate_jax_transformations():
    """Demonstrate powerful JAX transformations."""
    print("\n" + "=" * 60)
    print("JAX Transformations Demo")
    print("=" * 60)

    key = random.PRNGKey(123)
    X, y = generate_data(key, n_samples=100, n_features=5)
    params = init_mlp_params(key, [5, 32, 1])

    # 1. Vectorization with vmap
    print("\n1. Vectorized computation with vmap:")

    # Process each sample individually (inefficient)
    def process_single(x_single):
        return mlp_forward(params, x_single.reshape(1, -1))

    # Vectorize to process all samples at once
    vectorized_forward = jax.vmap(
        lambda x: mlp_forward(params, x.reshape(1, -1)).squeeze()
    )

    start_time = time.time()
    predictions_vmap = vectorized_forward(X)
    vmap_time = time.time() - start_time

    start_time = time.time()
    predictions_normal = mlp_forward(params, X)
    normal_time = time.time() - start_time

    print(f"vmap predictions shape: {predictions_vmap.shape}")
    print(f"Normal predictions shape: {predictions_normal.shape}")
    print(
        f"Results match: {jnp.allclose(predictions_vmap, predictions_normal.squeeze())}"
    )

    # 2. Automatic differentiation beyond gradients
    print("\n2. Higher-order derivatives:")

    def simple_function(x):
        return x**3 + 2 * x**2 + x

    # First derivative
    first_deriv = jax.grad(simple_function)
    # Second derivative
    second_deriv = jax.grad(first_deriv)

    x_test = 2.0
    print(f"f({x_test}) = {simple_function(x_test)}")
    print(f"f'({x_test}) = {first_deriv(x_test)}")
    print(f"f''({x_test}) = {second_deriv(x_test)}")

    # 3. JIT compilation benefits
    print("\n3. JIT compilation speedup:")

    # Compile the forward pass
    compiled_forward = jax.jit(mlp_forward)

    # Warm up
    _ = compiled_forward(params, X)
    _ = mlp_forward(params, X)

    # Benchmark
    n_runs = 100

    start_time = time.time()
    for _ in range(n_runs):
        _ = mlp_forward(params, X).block_until_ready()
    uncompiled_time = time.time() - start_time

    start_time = time.time()
    for _ in range(n_runs):
        _ = compiled_forward(params, X).block_until_ready()
    compiled_time = time.time() - start_time

    print(f"Uncompiled time: {uncompiled_time:.4f}s")
    print(f"Compiled time: {compiled_time:.4f}s")
    print(f"Speedup: {uncompiled_time / compiled_time:.2f}x")


if __name__ == "__main__":
    # Train the MLP
    trained_params, X_data, y_data = train_mlp()

    # Demonstrate JAX transformations
    demonstrate_jax_transformations()

    print("\n" + "=" * 60)
    print("Key JAX Concepts Demonstrated:")
    print("=" * 60)
    print("✓ Functional programming with immutable arrays")
    print("✓ Parameter initialization and management")
    print("✓ Forward pass implementation")
    print("✓ Automatic differentiation with jax.grad")
    print("✓ JIT compilation with jax.jit")
    print("✓ Vectorization with jax.vmap")
    print("✓ Training loop with gradient descent")
    print("✓ Higher-order derivatives")
    print("\nNext steps: Ready to build transformer components!")
