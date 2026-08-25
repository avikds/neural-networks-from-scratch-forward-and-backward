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

# Step 9 - make_optimizer
def make_optimizer(params, lr=1e-2, kind='sgd'):
    """Build an optimizer that updates params in place."""

    if lr <= 0:
        raise ValueError("lr must be positive")

    if kind != 'sgd':
        raise ValueError(f"Unsupported optimizer kind: {kind}")

    def step(grads):
        def update(param, grad):
            # Mutate the existing ndarray in place.
            param[...] -= lr * grad

        def walk(param_structure, grad_structure):
            if isinstance(param_structure, dict):
                for key in param_structure:
                    walk(param_structure[key], grad_structure[key])

            elif isinstance(param_structure, (list, tuple)):
                for p, g in zip(param_structure, grad_structure):
                    walk(p, g)

            elif isinstance(param_structure, np.ndarray):
                update(param_structure, grad_structure)

            else:
                raise TypeError(
                    f"Unsupported parameter type: {type(param_structure)}"
                )

        walk(params, grads)

    return {
        "step": step
    }

# Step 10 - train_step
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """Perform one complete optimization step over a minibatch.

    Returns the loss evaluated before the parameter update.
    """

    # Compute the current loss and gradients.
    loss, param_grads = forward_backward(
        model,
        loss_fn,
        x_batch,
        y_batch
    )

    # Apply exactly one in-place optimizer update.
    optimizer["step"](param_grads)

    # Return the pre-update loss.
    return float(loss)

# Step 11 - train
def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0):
    """Run a deterministic minibatch training loop."""

    if epochs < 0:
        raise ValueError("epochs must be non-negative")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    n = x.shape[0]

    if n == 0:
        return [float("nan")] * epochs

    # Create one reproducible RNG for the entire training run.
    rng = np.random.RandomState(seed)

    history = []

    for _ in range(epochs):
        # Shuffle the sample indices for this epoch.
        indices = rng.permutation(n)

        epoch_losses = []

        # Process the entire dataset in minibatches.
        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]

            x_batch = x[batch_indices]
            y_batch = y[batch_indices]

            # train_step returns the loss BEFORE this minibatch update.
            batch_loss = train_step(
                model,
                loss_fn,
                optimizer,
                x_batch,
                y_batch
            )

            epoch_losses.append(float(batch_loss))

        # Mean loss across all minibatches in this epoch.
        history.append(float(np.mean(epoch_losses)))

    return history

# Step 12 - design_network
def design_network(input_dim, num_classes, seed=0):
    """Design and train a feedforward network on a nonlinear task."""

    if input_dim < 1:
        raise ValueError("input_dim must be at least 1")

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    # ------------------------------------------------------------
    # 1. Deterministic setup
    # ------------------------------------------------------------
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # ------------------------------------------------------------
    # 2. Generate a genuinely nonlinear classification dataset.
    #
    # For input_dim >= 2, the class is determined by x0 * x1.
    # The sign and magnitude of a product cannot be represented well
    # by a linear decision boundary.
    #
    # For input_dim == 1, use |x0| instead.
    # ------------------------------------------------------------
    n_samples = max(800, num_classes * 200)

    x = rng.uniform(-1.0, 1.0, size=(n_samples, input_dim))

    if input_dim == 1:
        signal = np.abs(x[:, 0])

        # Equal-width nonlinear bands in |x|.
        scaled = signal * num_classes
        y = np.floor(scaled).astype(int)
        y = np.minimum(y, num_classes - 1)

    else:
        product = x[:, 0] * x[:, 1]

        # Quantile thresholds produce approximately balanced classes.
        thresholds = np.quantile(
            product,
            np.linspace(0.0, 1.0, num_classes + 1)
        )

        y = np.digitize(
            product,
            thresholds[1:-1],
            right=False
        ).astype(int)

        # Keep additional dimensions as weak nuisance features.
        if input_dim > 2:
            x[:, 2:] *= 0.10

    # ------------------------------------------------------------
    # 3. Standardize features for stable optimization.
    # ------------------------------------------------------------
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)

    # Avoid division by zero for degenerate dimensions.
    std = np.where(std < 1e-8, 1.0, std)

    x = (x - mean) / std

    # ------------------------------------------------------------
    # 4. Build a nonlinear network.
    # ------------------------------------------------------------
    def init_fn(in_dim, out_dim):
        return initialize_weights(
            in_dim,
            out_dim,
            scheme='he'
        )

    model = make_sequential([
        make_dense(input_dim, 64, init_fn),
        make_activation('relu'),
        make_dense(64, 64, init_fn),
        make_activation('relu'),
        make_dense(64, num_classes, init_fn),
    ])

    # ------------------------------------------------------------
    # 5. Loss and optimizer.
    # ------------------------------------------------------------
    loss_fn = make_loss('cross_entropy')

    optimizer = make_optimizer(
        model["params"],
        lr=0.01,
        kind='sgd'
    )

    # ------------------------------------------------------------
    # 6. Train.
    # ------------------------------------------------------------
    train(
        model,
        loss_fn,
        optimizer,
        x,
        y,
        epochs=500,
        batch_size=64,
        seed=seed
    )

    # ------------------------------------------------------------
    # 7. Evaluate on the exact dataset returned in metrics.
    # ------------------------------------------------------------
    logits, _ = model["forward"](x)

    predictions = np.argmax(logits, axis=1)

    accuracy = float(
        np.mean(predictions == y)
    )

    metrics = {
        "accuracy": accuracy,
        "x": x,
        "y": y
    }

    return model, metrics

# Step 13 - improve_generalization
import numpy as np


def improve_generalization(
    baseline_model_fn,
    x_train,
    y_train,
    x_val,
    y_val,
    seed=0
):
    """Improve held-out accuracy over an unregularized baseline."""

    np.random.seed(seed)

    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    x_val = np.asarray(x_val, dtype=float)
    y_val = np.asarray(y_val, dtype=int)

    if x_train.ndim != 2 or x_val.ndim != 2:
        raise ValueError("Training and validation inputs must be 2-D")

    if y_train.ndim != 1 or y_val.ndim != 1:
        raise ValueError("Training and validation labels must be 1-D")

    if len(x_train) != len(y_train):
        raise ValueError("x_train and y_train must have the same length")

    if len(x_val) != len(y_val):
        raise ValueError("x_val and y_val must have the same length")

    # ------------------------------------------------------------
    # Helper: evaluate a model.
    # ------------------------------------------------------------
    def evaluate(model):
        logits, _ = model["forward"](x_val)
        predictions = np.argmax(logits, axis=1)
        accuracy = float(np.mean(predictions == y_val))
        return accuracy, predictions

    # ------------------------------------------------------------
    # 1. Train the plain unregularized SGD baseline.
    # ------------------------------------------------------------
    baseline_model = baseline_model_fn()

    baseline_loss_fn = make_loss("cross_entropy")

    baseline_optimizer = make_optimizer(
        baseline_model["params"],
        lr=0.01,
        kind="sgd"
    )

    # Deliberately plain SGD:
    # no weight decay, no early stopping, no augmentation.
    train(
        baseline_model,
        baseline_loss_fn,
        baseline_optimizer,
        x_train,
        y_train,
        epochs=100,
        batch_size=min(32, len(x_train)),
        seed=seed
    )

    baseline_val_accuracy, _ = evaluate(baseline_model)

    # ------------------------------------------------------------
    # 2. Fresh model for the improved training setup.
    # ------------------------------------------------------------
    np.random.seed(seed)
    improved_model = baseline_model_fn()

    loss_fn = make_loss("cross_entropy")

    optimizer = make_optimizer(
        improved_model["params"],
        lr=0.005,
        kind="sgd"
    )

    # ------------------------------------------------------------
    # Helpers for copying/restoring the model parameters.
    # The arrays themselves are never replaced; only their contents
    # are copied so parameter identity is preserved.
    # ------------------------------------------------------------
    def copy_params(params):
        if isinstance(params, dict):
            return {
                key: copy_params(value)
                for key, value in params.items()
            }

        if isinstance(params, list):
            return [copy_params(value) for value in params]

        if isinstance(params, tuple):
            return tuple(copy_params(value) for value in params)

        if isinstance(params, np.ndarray):
            return params.copy()

        raise TypeError(f"Unsupported parameter type: {type(params)}")

    def restore_params(params, saved):
        if isinstance(params, dict):
            for key in params:
                restore_params(params[key], saved[key])

        elif isinstance(params, (list, tuple)):
            for p, s in zip(params, saved):
                restore_params(p, s)

        elif isinstance(params, np.ndarray):
            params[...] = saved

        else:
            raise TypeError(f"Unsupported parameter type: {type(params)}")

    # ------------------------------------------------------------
    # 3. L2 regularization.
    #
    # Add lambda * W to each parameter gradient. Biases are left
    # unregularized, which is a standard practical choice.
    # ------------------------------------------------------------
    weight_decay = 1e-4

    def add_weight_decay(grads, params):
        if isinstance(params, dict):
            for key in params:
                if key not in grads:
                    continue

                if key == "W":
                    grads[key][...] += weight_decay * params[key]

        elif isinstance(params, (list, tuple)):
            for p, g in zip(params, grads):
                add_weight_decay(g, p)

    # ------------------------------------------------------------
    # 4. Train with early stopping.
    #
    # Validation accuracy is used only to select the best trained
    # model; predictions are always generated from that model.
    # ------------------------------------------------------------
    rng = np.random.RandomState(seed)

    best_accuracy = -np.inf
    best_params = copy_params(improved_model["params"])

    patience = 20
    epochs_without_improvement = 0
    max_epochs = 150
    batch_size = min(32, len(x_train))

    for _ in range(max_epochs):
        indices = rng.permutation(len(x_train))

        for start in range(0, len(x_train), batch_size):
            batch_indices = indices[start:start + batch_size]

            xb = x_train[batch_indices]
            yb = y_train[batch_indices]

            loss, grads = forward_backward(
                improved_model,
                loss_fn,
                xb,
                yb
            )

            add_weight_decay(
                grads,
                improved_model["params"]
            )

            optimizer["step"](grads)

        # Evaluate after the epoch.
        val_accuracy, _ = evaluate(improved_model)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = copy_params(improved_model["params"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    # Restore the best validation checkpoint in place.
    restore_params(
        improved_model["params"],
        best_params
    )

    # ------------------------------------------------------------
    # 5. Generate the final predictions directly from the restored
    # improved model.
    # ------------------------------------------------------------
    logits, _ = improved_model["forward"](x_val)

    predictions = np.argmax(logits, axis=1).astype(int)

    val_accuracy = float(
        np.mean(predictions == y_val)
    )

    # ------------------------------------------------------------
    # 6. Return the actual measured results.
    # ------------------------------------------------------------
    return {
        "val_accuracy": val_accuracy,
        "baseline_val_accuracy": float(baseline_val_accuracy),
        "predictions": predictions,
        "model": improved_model
    }

