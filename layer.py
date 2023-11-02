import numpy as np
from activation import sigmoid



class Layer:
    def __init__(self, input_size, output_size, activation_function=sigmoid):
        self.weights = np.random.randn(
            input_size, output_size).astype(dtype=np.float32)
        self.bias = np.random.randn(output_size).astype(dtype=np.float32)
        self.activation_function = activation_function
        self.weights_gradient = None
        self.bias_gradient = None
        self.input = None

    def forward(self, values):
        self.input = values
        return np.matmul(values, self.weights) + self.bias

    def backwards(self, values, loss):
        delta = self.activation_function(
            values, compute_derivative=True) * loss
        self.bias_gradient = np.sum(delta, axis=0)
        self.weights_gradient = np.matmul(self.input.T, delta)
        return np.matmul(delta, self.weights.T)

    def activate(self, values):
        return self.activation_function(values)

    def updateWeights(self, learning_rate):
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * self.bias_gradient
