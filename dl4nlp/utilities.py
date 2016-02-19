import random
import numpy as np


def softmax(x):
    """
    Normalize any vector to probabilistic distribution.
    :param x: numpy array or matrix
    :return: numpy array or matrix of the same shape to x
    """
    xmax = np.expand_dims(np.max(x, -1), -1)
    e_x = np.exp(x - xmax)
    x = e_x / np.expand_dims(np.sum(e_x, -1), -1)
    return x


def sigmoid(x):
    """
    Sigmoid function
    :param x: scalar value
    :return: probability value
    """
    x = 1.0 / (1.0 + np.exp(-x))
    return x


def sigmoid_grad(f):
    """
    Sigmoid gradient function
    :param f: function value of sigmoid function
    :return: gradient value of sigmoid function
    """
    grad = f * (1.0 - f)
    return grad


def gradient_check_internal(f, x, h=1e-4, threshold=1e-5):
    """
    Gradient check for a function f
    :param f: a function that takes a single argument and outputs the cost and its gradients
    :param x: the point (numpy array) to check the gradient at
    :param h: small number to compute numerical gradient (optional)
    :param threshold: threshold to fail gradient check (optional)
    :return: list of tuple about information at failed point (empty list if passed)
    """
    rndstate = random.getstate()
    random.setstate(rndstate)
    np.random.seed(0)
    fx, grad = f(x)     # Evaluate function value at original point

    # Iterate over all indices in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    result = []
    while not it.finished:
        ix = it.multi_index

        x_step = x.copy()
        x_step[ix] += h / 2
        random.setstate(rndstate)
        np.random.seed(0)
        fx_pos, _ = f(x_step)

        x_step = x.copy()
        x_step[ix] -= h / 2
        random.setstate(rndstate)
        np.random.seed(0)
        fx_neg, _ = f(x_step)

        numgrad = (fx_pos - fx_neg) / h

        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))

        if reldiff > threshold:
            result.append((ix, grad[ix], numgrad))

        it.iternext()   # Step to next dimension

    return result


def gradient_check(f, x, h=1e-4, threshold=1e-5):
    """
    Gradient check for a function f
    :param f: a function that takes a single argument and outputs the cost and its gradients
    :param x: the point (numpy array) to check the gradient at
    :param h: small number to compute numerical gradient (optional)
    :param threshold: threshold to fail gradient check (optional)
    :return: true if it passed gradient check, false if failed.
    """
    result = gradient_check_internal(f, x, h, threshold)
    return len(result) == 0
