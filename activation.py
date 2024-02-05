import numpy as np


def sigmoid(A, compute_derivative=False):
    """Sigmoid activation function, element-wise (or its derivative with respect to its argument)."""

    if not compute_derivative:
        return 1.0 / (1.0 + np.exp(-A))  # element-wise operations
    else:
        # element-wise (Hadamard) product
        return sigmoid(A) * (1.0 - sigmoid(A))


def ReLU(A, compute_derivative=False):
    if not compute_derivative:
        return np.maximum(0, A)
    else:
        return np.where(A > 0, 1, 0)
