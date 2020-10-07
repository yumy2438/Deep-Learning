import numpy as np


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z)), Z


def relu(Z):
    return np.maximum(0, Z), Z


def sigmoid_backward(dA, Z):
    sigmoid_z,Z = sigmoid(Z)
    return dA * sigmoid_z * (1 - sigmoid_z) # A-Y


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def linear_activation_forward(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b
    assert (Z.shape == (W.shape[0], A_prev.shape[1]))
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    cache = (A_prev, W, activation_cache)
    return A, cache


def initialize_parameters_Xavier(layer_dims):
    parameters = {}

    for l in range(1, len(layer_dims)):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def initialize_parameters_He(layer_dims):
    parameters = {}

    for l in range(1, len(layer_dims)):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  # 1 w, 1 b
    for l in range(1, L):
        A, cache = linear_activation_forward(A, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)
    A_last, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    return A_last, caches


def compute_cost(AL, Y):
    "compute cross-entropy cost"
    m = Y.shape[1]
    cost = (-1 / m) * ((np.dot(Y, np.log(AL).T)) + (np.dot(1 - Y, np.log(1 - AL).T)))
    return cost


def compute_cost_with_regularization(AL, Y, parameters, l2_lambda):
    cross_entropy_cost = compute_cost(AL, Y)
    L = len(parameters) // 2
    m = Y.shape[1]
    l2_cost = 0
    for l in range(1, L + 1):
        l2_cost += np.sum(np.square(parameters['W' + str(l)]))
    l2_reg_cost = (1 / m) * (l2_lambda / 2) * l2_cost
    cost = cross_entropy_cost + l2_reg_cost
    return cost


def linear_backward(dZ, cache, l2_lambda):
    A_prev, W = cache
    m = A_prev.shape[1]
    dW = (1 / m) * np.dot(dZ, A_prev.T) + (l2_lambda/m)*W
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters


def linear_activation_backward(dA, cache, activation, l2_lambda):
    A, W, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    return linear_backward(dZ, (A, W), l2_lambda)


def model_backward(A_last, Y, caches, l2_lambda):
    grads = {}
    L = len(caches)
    Y = Y.reshape(A_last.shape)

    dA_last = -(np.divide(Y, A_last) - np.divide(1 - Y, 1 - A_last))

    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dA_last,
                                                                                                      current_cache,
                                                                                                      "sigmoid",l2_lambda)
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu", l2_lambda)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads