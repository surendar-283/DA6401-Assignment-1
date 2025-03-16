import numpy as np
import wandb
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh_derivative(x):
    return 1 - np.power(x, 2)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def identity(x):
    return x

def identity_derivative(x):
    return np.ones_like(x)

def initialize_parameters(layer_sizes, init_method="random", weight_decay=0):
    np.random.seed(1)
    p = []

    for i in range(len(layer_sizes) - 1):
        if init_method == "random":
            W = np.random.uniform(-0.1, 0.1, (layer_sizes[i], layer_sizes[i+1]))
        elif init_method == "xavier":
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / (layer_sizes[i] + layer_sizes[i+1]))

        b = np.zeros((1, layer_sizes[i+1]))
        p.append((W, b))

    return p

def forward(X, p, activation="sigmoid"):
    a = {0: X}
    A = X

    act_func = sigmoid
    if activation == "tanh":
        act_func = tanh
    elif activation == "relu":
        act_func = relu
    elif activation == "identity":
        act_func = identity

    for i, (W, b) in enumerate(p[:-1]):
        Z = np.dot(A, W) + b
        A = act_func(Z)
        a[i+1] = A

    W_out, b_out = p[-1]
    Z_out = np.dot(A, W_out) + b_out
    A_out = softmax(Z_out)

    a[len(p)] = A_out
    return A_out, a

def backward(X, Y, p, a, activation="sigmoid", wd=0, loss_type="cross_entropy"):
    m = X.shape[0]
    gradients = {}

    act_deriv = sigmoid_derivative
    if activation == "tanh":
        act_deriv = tanh_derivative
    elif activation == "relu":
        act_deriv = relu_derivative
    elif activation == "identity":
        act_deriv = identity_derivative

    A_out = a[len(p)]
    
    if loss_type == "cross_entropy":
        dZ_out = A_out.copy()
        dZ_out[np.arange(m), Y] -= 1
        dZ_out /= m
    elif loss_type == "mse":
        Y_one_hot = np.zeros_like(A_out)
        Y_one_hot[np.arange(m), Y] = 1
        dZ_out = 2 * (A_out - Y_one_hot) / m

    W_out = p[-1][0]
    reg_term = 0
    if wd > 0:
        reg_term = wd * W_out

    gradients[len(p)] = (np.dot(a[len(p)-1].T, dZ_out) + reg_term,
                          np.sum(dZ_out, axis=0, keepdims=True))

    dA = np.dot(dZ_out, p[-1][0].T)

    for i in reversed(range(len(p) - 1)):
        if activation == "relu":
            if i == 0:
                prev_A = a[i]
            else:
                prev_A = a[i]

            W, b = p[i]
            Z = np.dot(prev_A, W) + b
            dZ = dA * relu_derivative(Z)
        elif activation == "identity":
            dZ = dA * identity_derivative(a[i+1])
        else:
            dZ = dA * act_deriv(a[i+1])

        W = p[i][0]
        reg_term = 0
        if wd > 0:
            reg_term = wd * W

        gradients[i+1] = (np.dot(a[i].T, dZ) + reg_term,
                          np.sum(dZ, axis=0, keepdims=True))

        dA = np.dot(dZ, p[i][0].T)

    return gradients

def compute_loss(Y_pred, Y_true, p=None, wd=0, loss_type="cross_entropy"):
    m = Y_true.shape[0]
    
    if loss_type == "cross_entropy":
        cross_entropy = -np.mean(np.log(Y_pred[np.arange(m), Y_true] + 1e-9))
        loss = cross_entropy
    elif loss_type == "mse":
        Y_true_one_hot = np.zeros_like(Y_pred)
        Y_true_one_hot[np.arange(m), Y_true] = 1
        loss = np.mean(np.sum((Y_pred - Y_true_one_hot) ** 2, axis=1))
    reg_term = 0
    if wd > 0 and p is not None:
        for W, _ in p:
            reg_term += 0.5 * wd * np.sum(np.square(W))

    return loss + reg_term

def update_parameters(p, g, lr):
    for i in range(len(p)):
        W, b = p[i]
        dW, db = g[i+1]
        W = W - lr * dW
        b = b - lr * db
        p[i] = (W, b)
    return p

def create_train_val_split(X, y, val_ratio=0.1, random_seed=1):
    np.random.seed(random_seed)
    indices = np.random.permutation(len(X))
    val_size = int(len(X) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]

    return X_train, X_val, y_train, y_val

def sgd(config, X_val, y_val, X_train, y_train, parameters):
    batch_size = config.batch_size
    weight_decay = config.weight_decay
    loss_type = config.loss_type if hasattr(config, 'loss_type') else "cross_entropy"
    
    for epoch in range(config.epochs):
        batch_losses = []
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            y_pred, layer_activations = forward(X_batch, parameters, activation=config.activation)

            gradients = backward(X_batch, y_batch, parameters, layer_activations,activation=config.activation, wd=weight_decay, loss_type=loss_type)

            parameters = update_parameters(parameters, gradients, config.learning_rate)

            batch_loss = compute_loss(y_pred, y_batch, parameters, weight_decay, loss_type=loss_type)
            batch_losses.append(batch_loss)

        epoch_loss = np.mean(batch_losses)

        train_pred, _ = forward(X_train, parameters, activation=config.activation)
        train_acc = np.mean(np.argmax(train_pred, axis=1) == y_train)

        val_pred, _ = forward(X_val, parameters, activation=config.activation)
        val_loss = compute_loss(val_pred, y_val, parameters, weight_decay, loss_type=loss_type)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)
        
        print(f"Epoch {epoch+1}/{config.epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

def momentum(config, X_val, y_val, X_train, y_train, parameters):
    batch_size = config.batch_size
    weight_decay = config.weight_decay
    loss_type = config.loss_type if hasattr(config, 'loss_type') else "cross_entropy"
    momentum_values = {i: (np.zeros_like(W), np.zeros_like(b)) for i, (W, b) in enumerate(parameters)}
    beta = config.momentum

    for epoch in range(config.epochs):
        indices = np.random.permutation(X_train.shape[0])
        xs = X_train[indices]
        ys = y_train[indices]

        batch_losses = []

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = xs[i:i + batch_size]
            y_batch = ys[i:i + batch_size]

            y_pred, layer_activations = forward(X_batch, parameters, activation=config.activation)

            gradients = backward(X_batch, y_batch, parameters, layer_activations,activation=config.activation, wd=weight_decay, loss_type=loss_type)

            for j in range(len(parameters)):
                W, b = parameters[j]
                dW, db = gradients[j + 1]
                vW, vb = momentum_values[j]

                vW = beta * vW - config.learning_rate * dW
                vb = beta * vb - config.learning_rate * db

                W = W + vW
                b = b + vb

                parameters[j] = (W, b)
                momentum_values[j] = (vW, vb)

            batch_loss = compute_loss(y_pred, y_batch, parameters, weight_decay, loss_type=loss_type)
            batch_losses.append(batch_loss)

        epoch_loss = np.mean(batch_losses)

        train_pred, _ = forward(X_train, parameters, activation=config.activation)
        train_acc = np.mean(np.argmax(train_pred, axis=1) == y_train)

        val_pred, _ = forward(X_val, parameters, activation=config.activation)
        val_loss = compute_loss(val_pred, y_val, parameters, weight_decay, loss_type=loss_type)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)

        print(f"Epoch {epoch+1}/{config.epochs}, Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

def nesterov(config, X_val, y_val, X_train, y_train, parameters):
    batch_size = config.batch_size
    weight_decay = config.weight_decay
    loss_type = config.loss_type if hasattr(config, 'loss_type') else "cross_entropy"
    momentum_values = {i: (np.zeros_like(W), np.zeros_like(b)) for i, (W, b) in enumerate(parameters)}
    beta = config.momentum

    for epoch in range(config.epochs):
        indices = np.random.permutation(X_train.shape[0])
        xs = X_train[indices]
        ys = y_train[indices]

        batch_losses = []

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = xs[i:i + batch_size]
            y_batch = ys[i:i + batch_size]

            lookahead_parameters = []
            for j in range(len(parameters)):
                W, b = parameters[j]
                vW, vb = momentum_values[j]
                lookahead_W = W + beta * vW
                lookahead_b = b + beta * vb
                lookahead_parameters.append((lookahead_W, lookahead_b))

            y_pred, layer_activations = forward(X_batch, lookahead_parameters, activation=config.activation)

            gradients = backward(X_batch, y_batch, lookahead_parameters, layer_activations,activation=config.activation, wd=weight_decay, loss_type=loss_type)

            for j in range(len(parameters)):
                W, b = parameters[j]
                dW, db = gradients[j + 1]
                vW, vb = momentum_values[j]

                vW = beta * vW - config.learning_rate * dW
                vb = beta * vb - config.learning_rate * db

                W = W + vW
                b = b + vb

                parameters[j] = (W, b)
                momentum_values[j] = (vW, vb)

            batch_loss = compute_loss(y_pred, y_batch, parameters, weight_decay, loss_type=loss_type)
            batch_losses.append(batch_loss)

        epoch_loss = np.mean(batch_losses)

        train_pred, _ = forward(X_train, parameters, activation=config.activation)
        train_acc = np.mean(np.argmax(train_pred, axis=1) == y_train)

        val_pred, _ = forward(X_val, parameters, activation=config.activation)
        val_loss = compute_loss(val_pred, y_val, parameters, weight_decay, loss_type=loss_type)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)

        print(f"Epoch {epoch+1}/{config.epochs}, Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

def rmsprop(config, X_val, y_val, X_train, y_train, parameters):
    batch_size = config.batch_size
    weight_decay = config.weight_decay
    loss_type = config.loss_type if hasattr(config, 'loss_type') else "cross_entropy"
    squared_grads = {i: (np.zeros_like(W), np.zeros_like(b)) for i, (W, b) in enumerate(parameters)}
    beta = config.beta 
    epsilon = config.epsilon


    for epoch in range(config.epochs):
        indices = np.random.permutation(X_train.shape[0])
        xs = X_train[indices]
        ys = y_train[indices]

        batch_losses = []

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = xs[i:i + batch_size]
            y_batch = ys[i:i + batch_size]

            y_pred, layer_activations = forward(X_batch, parameters, activation=config.activation)

            gradients = backward(X_batch, y_batch, parameters, layer_activations,activation=config.activation, wd=weight_decay, loss_type=loss_type)

            for j in range(len(parameters)):
                W, b = parameters[j]
                dW, db = gradients[j + 1]
                sW, sb = squared_grads[j]

                sW = beta * sW + (1 - beta) * (dW ** 2)
                sb = beta * sb + (1 - beta) * (db ** 2)

                W = W - config.learning_rate * dW / (np.sqrt(sW) + epsilon)
                b = b - config.learning_rate * db / (np.sqrt(sb) + epsilon)

                parameters[j] = (W, b)
                squared_grads[j] = (sW, sb)

            batch_loss = compute_loss(y_pred, y_batch, parameters, weight_decay, loss_type=loss_type)
            batch_losses.append(batch_loss)

        epoch_loss = np.mean(batch_losses)

        train_pred, _ = forward(X_train, parameters, activation=config.activation)
        train_acc = np.mean(np.argmax(train_pred, axis=1) == y_train)

        val_pred, _ = forward(X_val, parameters, activation=config.activation)
        val_loss = compute_loss(val_pred, y_val, parameters, weight_decay, loss_type=loss_type)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)

        print(f"Epoch {epoch+1}/{config.epochs}, Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

def adam(config, X_val, y_val, X_train, y_train, parameters):
    batch_size = config.batch_size
    weight_decay = config.weight_decay
    loss_type = config.loss_type if hasattr(config, 'loss_type') else "cross_entropy"
    first_moment = {i: (np.zeros_like(W), np.zeros_like(b)) for i, (W, b) in enumerate(parameters)}
    second_moment = {i: (np.zeros_like(W), np.zeros_like(b)) for i, (W, b) in enumerate(parameters)}
    beta1 = config.beta1  
    beta2 = config.beta2 
    epsilon = config.epsilon
    t = 0

    for epoch in range(config.epochs):
        indices = np.random.permutation(X_train.shape[0])
        xs = X_train[indices]
        ys = y_train[indices]

        batch_losses = []

        for i in range(0, X_train.shape[0], batch_size):
            t += 1
            X_batch = xs[i:i + batch_size]
            y_batch = ys[i:i + batch_size]

            y_pred, layer_activations = forward(X_batch, parameters, activation=config.activation)

            gradients = backward(X_batch, y_batch, parameters, layer_activations,activation=config.activation, wd=weight_decay, loss_type=loss_type)

            for j in range(len(parameters)):
                W, b = parameters[j]
                dW, db = gradients[j + 1]
                mW, mb = first_moment[j]
                vW, vb = second_moment[j]

                mW = beta1 * mW + (1 - beta1) * dW
                mb = beta1 * mb + (1 - beta1) * db

                vW = beta2 * vW + (1 - beta2) * (dW ** 2)
                vb = beta2 * vb + (1 - beta2) * (db ** 2)

                mW_hat = mW / (1 - beta1 ** t)
                mb_hat = mb / (1 - beta1 ** t)

                vW_hat = vW / (1 - beta2 ** t)
                vb_hat = vb / (1 - beta2 ** t)

                W = W - config.learning_rate * mW_hat / (np.sqrt(vW_hat) + epsilon)
                b = b - config.learning_rate * mb_hat / (np.sqrt(vb_hat) + epsilon)

                parameters[j] = (W, b)
                first_moment[j] = (mW, mb)
                second_moment[j] = (vW, vb)

            batch_loss = compute_loss(y_pred, y_batch, parameters, weight_decay, loss_type=loss_type)
            batch_losses.append(batch_loss)

        epoch_loss = np.mean(batch_losses)

        train_pred, _ = forward(X_train, parameters, activation=config.activation)
        train_acc = np.mean(np.argmax(train_pred, axis=1) == y_train)

        val_pred, _ = forward(X_val, parameters, activation=config.activation)
        val_loss = compute_loss(val_pred, y_val, parameters, weight_decay, loss_type=loss_type)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)

        print(f"Epoch {epoch+1}/{config.epochs}, Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

def nadam(config, X_val, y_val, X_train, y_train, parameters):
    batch_size = config.batch_size
    weight_decay = config.weight_decay
    loss_type = config.loss_type if hasattr(config, 'loss_type') else "cross_entropy"
    first_moment = {i: (np.zeros_like(W), np.zeros_like(b)) for i, (W, b) in enumerate(parameters)}
    second_moment = {i: (np.zeros_like(W), np.zeros_like(b)) for i, (W, b) in enumerate(parameters)}
    beta1 = config.beta1
    beta2 = config.beta2 
    epsilon = config.epsilon
    t = 0

    for epoch in range(config.epochs):
        indices = np.random.permutation(X_train.shape[0])
        xs = X_train[indices]
        ys = y_train[indices]

        batch_losses = []

        for i in range(0, X_train.shape[0], batch_size):
            t += 1
            X_batch = xs[i:i + batch_size]
            y_batch = ys[i:i + batch_size]

            y_pred, layer_activations = forward(X_batch, parameters, activation=config.activation)

            gradients = backward(X_batch, y_batch, parameters, layer_activations,activation=config.activation, wd=weight_decay, loss_type=loss_type)

            for j in range(len(parameters)):
                W, b = parameters[j]
                dW, db = gradients[j + 1]
                mW, mb = first_moment[j]
                vW, vb = second_moment[j]

                mW = beta1 * mW + (1 - beta1) * dW
                mb = beta1 * mb + (1 - beta1) * db

                vW = beta2 * vW + (1 - beta2) * (dW ** 2)
                vb = beta2 * vb + (1 - beta2) * (db ** 2)

                mW_hat = mW / (1 - beta1 ** t)
                mb_hat = mb / (1 - beta1 ** t)

                vW_hat = vW / (1 - beta2 ** t)
                vb_hat = vb / (1 - beta2 ** t)

                mW_nesterov = beta1 * mW_hat + (1 - beta1) * dW / (1 - beta1 ** t)
                mb_nesterov = beta1 * mb_hat + (1 - beta1) * db / (1 - beta1 ** t)

                W = W - config.learning_rate * mW_nesterov / (np.sqrt(vW_hat) + epsilon)
                b = b - config.learning_rate * mb_nesterov / (np.sqrt(vb_hat) + epsilon)

                parameters[j] = (W, b)
                first_moment[j] = (mW, mb)
                second_moment[j] = (vW, vb)

            batch_loss = compute_loss(y_pred, y_batch, parameters, weight_decay, loss_type=loss_type)
            batch_losses.append(batch_loss)

        epoch_loss = np.mean(batch_losses)

        train_pred, _ = forward(X_train, parameters, activation=config.activation)
        train_acc = np.mean(np.argmax(train_pred, axis=1) == y_train)

        val_pred, _ = forward(X_val, parameters, activation=config.activation)
        val_loss = compute_loss(val_pred, y_val, parameters, weight_decay, loss_type=loss_type)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)

        print(f"Epoch {epoch+1}/{config.epochs}, Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
