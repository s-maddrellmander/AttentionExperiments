import jax
import jax.numpy as jnp
from jax import random, jit
from jax.scipy.special import logsumexp
from jax import lax


def softmax(x):
    return jnp.exp(x - logsumexp(x, axis=-1, keepdims=True))


def scaled_dot_product_attention(Q, K, V, return_scores=False):
    d_k = Q.shape[-1]
    scores = jnp.matmul(Q, K.transpose(-1, -2)) / jnp.sqrt(d_k)
    weights = softmax(scores)
    output = jnp.matmul(weights, V)
    if return_scores:
        return output, weights, scores  # return both output and attention weights
    return output


def sliding_window_attention(Q, K, V, window_size, sparse=False):
    if sparse is False:
        d_k = Q.shape[-1]
        scores = jnp.matmul(Q, K.transpose(-1, -2)) / jnp.sqrt(d_k)

        # Create a mask for values outside the sliding window
        max_seq_len = Q.shape[0]
        indices = jnp.arange(max_seq_len).reshape(-1, 1)
        mask = jnp.abs(indices - indices.T) > window_size // 2
        scores = jnp.where(mask, jnp.float32(-1e9), scores)

        weights = softmax(scores)
        return jnp.matmul(weights, V), scores

    else:
        d_k = Q.shape[-1]
        max_seq_len = Q.shape[0]

        outputs = []

        for i in range(max_seq_len):
            # Compute the start and end of the current window
            start = jnp.maximum(0, i - window_size // 2)
            end = jnp.minimum(max_seq_len, i + window_size // 2 + 1)
            window_size_actual = end - start

            # Use dynamic_slice to extract the relevant portion of K and V
            relevant_K = lax.dynamic_slice(K, (start, 0), (window_size_actual, d_k))
            relevant_V = lax.dynamic_slice(V, (start, 0), (window_size_actual, d_k))

            # Compute scores only for the window
            scores = jnp.matmul(Q[i : i + 1], relevant_K.transpose(-1, -2)) / jnp.sqrt(
                d_k
            )

            # Compute weights and output
            weights = softmax(scores)
            output = jnp.matmul(weights, relevant_V)
            outputs.append(output)

        return jnp.concatenate(outputs, axis=0)
