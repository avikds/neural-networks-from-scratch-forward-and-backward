"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
import numpy as np

def numerical_gradient(f, x, eps=1e-5):
    """Estimate the gradient of scalar f with respect to x
    using central finite differences.
    """
    grad = np.zeros_like(x, dtype=float)

    # Handle arrays of any shape, including empty arrays.
    for idx in np.ndindex(x.shape):
        original_value = x[idx]

        x[idx] = original_value + eps
        f_plus = f(x)

        x[idx] = original_value - eps
        f_minus = f(x)

        x[idx] = original_value

        grad[idx] = (f_plus - f_minus) / (2.0 * eps)

    return grad

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    """Return the maximum relative error between analytic and numeric gradients."""
    analytic_grad = np.asarray(analytic_grad, dtype=float)
    numeric_grad = np.asarray(numeric_grad, dtype=float)

    denominator = np.maximum(
        np.maximum(np.abs(analytic_grad), np.abs(numeric_grad)),
        tol
    )

    relative_error = np.abs(analytic_grad - numeric_grad) / denominator

    if relative_error.size == 0:
        return 0.0

    return float(np.max(relative_error))

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer."""
    W, b = weight_init_fn(in_dim, out_dim)

    params = {
        "W": W,
        "b": b
    }

    def forward(x):
        # Affine transformation: y = x @ W + b
        y = x @ params["W"] + params["b"]

        # Cache the input for the backward pass.
        cache = x

        return y, cache

    def backward(dout, cache):
        # cache is the input x from the forward pass.
        x = cache

        # Gradient with respect to input:
        # dx = dout @ W^T
        dx = dout @ params["W"].T

        # Gradient with respect to weights:
        # dW = x^T @ dout
        dW = x.T @ dout

        # Gradient with respect to bias:
        # Sum over the batch dimension.
        db = np.sum(dout, axis=0)

        grads = {
            "W": dW,
            "b": db
        }

        return dx, grads

    return {
        "params": params,
        "forward": forward,
        "backward": backward
    }

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer."""
    
    if kind != 'relu':
        raise ValueError(f"Unsupported activation kind: {kind}")

    params = {}

    def forward(x):
        # ReLU: max(0, x)
        y = np.maximum(0, x)

        # Cache x because the backward pass needs its sign.
        cache = x

        return y, cache

    def backward(dout, cache):
        x = cache

        # ReLU derivative:
        # 0 for x < 0, 1 for x > 0.
        # At x == 0, choosing 0 is the standard convention.
        dx = dout * (x > 0)

        return dx, {}

    return {
        "params": params,
        "forward": forward,
        "backward": backward
    }

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer."""
    
    if in_dim <= 0 or out_dim <= 0:
        raise ValueError("in_dim and out_dim must be positive")

    if scheme == 'he':
        # He initialization for ReLU networks:
        # std = sqrt(2 / fan_in)
        std = np.sqrt(2.0 / in_dim)
        W = np.random.randn(in_dim, out_dim) * std

    elif scheme == 'xavier':
        # Xavier/Glorot normal initialization
        std = np.sqrt(2.0 / (in_dim + out_dim))
        W = np.random.randn(in_dim, out_dim) * std

    else:
        raise ValueError(f"Unsupported initialization scheme: {scheme}")

    # Zero bias keeps the initial activations centered.
    b = np.zeros(out_dim, dtype=float)

    return W, b

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

