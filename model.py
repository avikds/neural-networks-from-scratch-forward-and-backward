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

    if kind not in ('relu', 'tanh'):
        raise ValueError(f"Unsupported activation kind: {kind}")

    params = {}

    def forward(x):
        if kind == 'relu':
            y = np.maximum(0, x)
        else:  # tanh
            y = np.tanh(x)

        # Cache the activation output for efficient backward computation.
        cache = y

        return y, cache

    def backward(dout, cache):
        y = cache

        if kind == 'relu':
            dx = dout * (y > 0)
        else:  # tanh
            dx = dout * (1.0 - y ** 2)

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

# Step 6 - make_loss
def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits)."""

    if kind != 'cross_entropy':
        raise ValueError(f"Unsupported loss kind: {kind}")

    def loss_fn(logits, labels):
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=int)

        batch_size = logits.shape[0]

        # Numerically stable log-sum-exp:
        # log(sum(exp(logits))) = max(logits) + log(sum(exp(logits - max)))
        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted = logits - max_logits

        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)

        log_sum_exp = np.log(sum_exp) + max_logits

        # Cross-entropy:
        # L_i = log(sum(exp(logits_i))) - logits_i[label_i]
        correct_logits = logits[np.arange(batch_size), labels]
        per_example_loss = log_sum_exp[:, 0] - correct_logits

        loss = float(np.mean(per_example_loss))

        # Softmax probabilities.
        probs = exp_shifted / sum_exp

        # dL/dlogits = (softmax - one_hot(labels)) / batch_size
        d_logits = probs.copy()
        d_logits[np.arange(batch_size), labels] -= 1.0
        d_logits /= batch_size

        return loss, d_logits

    return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model."""

    def forward(x):
        caches = []
        out = x

        for layer in layers:
            out, cache = layer["forward"](out)
            caches.append(cache)

        return out, caches

    def backward(dout, caches):
        dx = dout
        grads_list = [None] * len(layers)

        for i in range(len(layers) - 1, -1, -1):
            dx, grads = layers[i]["backward"](dx, caches[i])
            grads_list[i] = grads

        return dx, grads_list

    return {
        "params": [layer["params"] for layer in layers],
        "forward": forward,
        "backward": backward
    }

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.

    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels

    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
    """

    # Forward pass through the complete model.
    logits, caches = model["forward"](x)

    # Compute loss and gradient with respect to the model output.
    loss, d_logits = loss_fn(logits, y)

    # Backward pass through the complete model.
    _, layer_grads = model["backward"](d_logits, caches)

    # Return gradients in the same per-layer structure as model["params"].
    return float(loss), layer_grads

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

