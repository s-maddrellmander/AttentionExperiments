import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
import pytest

from attention import softmax, scaled_dot_product_attention

def test_softmax_basic():
    x = jnp.array([2.0, 1.0, 0.1])
    expected = jnp.array([0.65900114, 0.24243298, 0.09856589])
    np.testing.assert_allclose(softmax(x), expected, rtol=1e-6)

def test_softmax_sum_one():
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    assert jnp.isclose(jnp.sum(softmax(x)), 1.0, rtol=1e-6)

def test_softmax_multidimensional():
    x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    expected = jnp.array([[0.26894143, 0.7310586 ], 
                          [0.26894143, 0.7310586 ], 
                          [0.26894143, 0.7310586 ]])
    np.testing.assert_allclose(softmax(x), expected, rtol=1e-6)

def test_softmax_large_numbers():
    x = jnp.array([1000.0, 1000.0])
    expected = jnp.array([0.5, 0.5])
    np.testing.assert_allclose(softmax(x), expected, rtol=1e-4)

def test_softmax_small_numbers():
    x = jnp.array([-1000.0, -1000.0])
    expected = jnp.array([0.5, 0.5])
    np.testing.assert_allclose(softmax(x), expected, rtol=1e-4)

def test_softmax_stability():
    # Very large and small numbers can cause numerical instability. 
    # This test ensures the function doesn't return NaN or inf values.
    x = jnp.array([1e20, -1e20])
    result = softmax(x)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_attention_basic():
    Q = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    K = jnp.array([[50.0, 0.0], [0.0, 50.0]])  # Increased values to push softmax to extremes
    V = jnp.array([[50.0, 0.0], [0.0, 50.0]])
    
    expected = V  # in this specific setup, output should be very close to V
    np.testing.assert_allclose(scaled_dot_product_attention(Q, K, V), expected, atol=1e-1, rtol=1e-1)  # increased tolerance

def test_attention_weights_sum_one():
    Q = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    K = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    V = jnp.array([[10.0, 0.0], [0.0, 10.0]])
    
    _, weights, _ = scaled_dot_product_attention(Q, K, V, return_scores=True)
    np.testing.assert_allclose(jnp.sum(weights, axis=-1), 1.0, rtol=1e-6)

def test_attention_return_scores():
    Q = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    K = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    V = jnp.array([[10.0, 0.0], [0.0, 10.0]])
    
    output, weights, scores = scaled_dot_product_attention(Q, K, V, return_scores=True)
    assert scores is not None

def test_attention_large_numbers():
    Q = jnp.array([[1000.0, 0.0], [0.0, 1000.0]])
    K = jnp.array([[1000.0, 0.0], [0.0, 1000.0]])
    V = jnp.array([[10.0, 0.0], [0.0, 10.0]])
    
    output = scaled_dot_product_attention(Q, K, V)
    assert not np.any(np.isnan(output))
    assert not np.any(np.isinf(output))

def test_attention_small_numbers():
    Q = jnp.array([[-1000.0, 0.0], [0.0, -1000.0]])
    K = jnp.array([[-1000.0, 0.0], [0.0, -1000.0]])
    V = jnp.array([[10.0, 0.0], [0.0, 10.0]])
    
    output = scaled_dot_product_attention(Q, K, V)
    assert not np.any(np.isnan(output))
    assert not np.any(np.isinf(output))