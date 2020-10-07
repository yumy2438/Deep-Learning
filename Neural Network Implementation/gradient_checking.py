from dnn_utils import *
import copy


def vector_to_dict(i, epsilon, original_params, sign):
    index = 0
    parameters = copy.deepcopy(original_params)
    for key in parameters:
        val = parameters[key]
        sh1, sh2 = val.shape
        index += sh1 * sh2
        if i < index:
            index -= sh1 * sh2
            nth = i - index
            row = nth // sh2
            column = nth % sh2
            if sign == '+':
                parameters[key][row, column] += epsilon
            else:
                parameters[key][row, column] -= epsilon
            break
    return parameters


def dict_to_vector(dict):
    first = True
    for key in dict:
        if 'A' not in key:  # only (d)W's and (d)b's
            vectorized = np.reshape(dict[key], (-1, 1))
            if first:
                answer = vectorized
                first = False
            else:
                answer = np.concatenate((answer, vectorized), axis=0)
    return answer


def get_params_orderly(length):
    length //= 3  # (a,w,b)
    params = []
    for x in range(length):
        params.append("dW" + str(x + 1))
        params.append("db" + str(x + 1))
    return params


def dict_to_vector_g(dict):
    first = True
    for key in get_params_orderly(len(dict.keys())):
        vectorized = np.reshape(dict[key], (-1, 1))
        if first:
            answer = vectorized
            first = False
        else:
            answer = np.concatenate((answer, vectorized), axis=0)
    return answer


def gradient_check(parameters, gradients, X, Y, epsilon=1e-7, l2_lambda=0):
    vectorized_params = dict_to_vector(parameters)
    vectorized_grads = dict_to_vector_g(gradients)
    num_params = vectorized_params.shape[0]
    J_plus = np.zeros((num_params, 1))
    J_minus = np.zeros((num_params, 1))
    gradapprox = np.zeros((num_params, 1))
    for i in range(num_params):
        dict_plus = vector_to_dict(i, epsilon, parameters, '+')
        A_last, _ = model_forward(X, dict_plus)
        J_plus[i] = compute_cost_with_regularization(A_last, Y, dict_plus, l2_lambda)

        dict_minus = vector_to_dict(i, epsilon, parameters, '-')
        A_last, _ = model_forward(X, dict_minus)
        J_minus[i] = compute_cost_with_regularization(A_last, Y, dict_minus, l2_lambda)
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    numerator = np.linalg.norm(vectorized_grads - gradapprox)
    denominator = np.linalg.norm(vectorized_grads) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
    if difference > 2e-7:
        print("difference:" + str(difference))
    else:
        print("Your implementation is correct!")

