from nn.layers import Layer
import numpy as np


class Linear(Layer):
    """Simple fully connected layer"""

    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self._reset_params()

    def _reset_params(self):
        self.W = 0.01 * np.random.rand(self.input_size, self.output_size)
        self.b = np.zeros((1, self.output_size))

    def forward(self, X):
        return np.dot(X, self.W) + self.b

    def backward(self, *input):
        pass
