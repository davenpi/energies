import time

import jax
import jax.numpy as jnp
import numpy as np

x_np = np.linspace(0, 10, 10)
x_jnp = jnp.linspace(0, 10, 10)

# mutate x_np
x_np[0] = 100

print(x_np)

# try to mutate x_jnp
try:
    x_jnp[0] = 100
except Exception as e:
    print(e)

print(isinstance(x_jnp, jax.Array))

print(x_jnp.devices())

print(x_jnp.sharding)


def norm(x):
    x = x - x.mean(0)
    return x / x.std(0)


norm_compiled = jax.jit(norm)

np.random.seed(11)
X = jnp.array(np.random.rand(10000, 100))
print(np.allclose(norm(X), norm_compiled(X), atol=1e-6))


def benchmark_norm(n):
    X = jnp.array(np.random.rand(10000, 100))
    start = time.time()
    for _ in range(n):
        norm(X).block_until_ready()
    end = time.time()
    return end - start


def benchmark_norm_compiled(n):
    X = jnp.array(np.random.rand(10000, 100))
    start = time.time()
    for _ in range(n):
        norm_compiled(X).block_until_ready()
    end = time.time()
    return end - start


time_norm = benchmark_norm(100)
print(f"Time taken: {time_norm} seconds")
time_norm_compiled = benchmark_norm_compiled(100)
print(f"Time taken: {time_norm_compiled} seconds")

print(f"Speedup: {time_norm / time_norm_compiled}")


def get_negatives(x):
    return x[x < 0]


x = jnp.array(np.random.randn(10))
print(x)
print(get_negatives(x))

jitted_negatives = jax.jit(get_negatives)
try:
    jitted_negatives(x)
except Exception as e:
    print(e)


@jax.jit
def f(x, y):
    print("Running f():")
    print(f"  {x = }")
    print(f"  {y = }")
    result = jnp.dot(x + 1, y + 1)
    print(f"  {result = }")
    return result


x = np.random.randn(3, 4)
y = np.random.randn(4)
ans = f(x, y)
print(ans)
ans2 = f(x, y)
print(ans2)


def f(x, y):
    return jnp.dot(x + 1, y + 1)


jaxpr = jax.make_jaxpr(f)(x, y)
print(jaxpr)
