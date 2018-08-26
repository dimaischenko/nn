from nn.modules import Module
import numpy as np


class Linear(Module):
    """Simple fully connected layer"""

    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self._reset_params()

    def _reset_params(self):
        self.W = 0.01 * np.random.rand(self.input_size, self.output_size)
        self.b = np.zeros((1, self.output_size))

    def forward(self, input_):
        self.output = np.dot(input_, self.W) + self.b
        return self.output

    def backward(self, grad_output):
        self.grad_input = np.dot(grad_output, (self.W).T)
        return self.grad_input
