from nn.modules import Module
import numpy as np


class Linear(Module):
    """Simple fully connected layer"""

    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self._init_params()

    def _init_params(self):
        self.params = [
            0.01 * np.random.rand(self.input_size, self.output_size),  # W
            np.zeros((1, self.output_size))  # b
        ]

        self.grad_params = [0, 0]

    def forward(self, input_):
        self.output = np.dot(input_, self.params[0]) + self.params[1]
        return self.output

    def backward(self, grad_output):
        self.grad_input = np.dot(grad_output, (self.params[0]).T)
        return self.grad_input

    def zero_grad(self):
        self.grad_params[0] = 0
        self.grad_params[1] = 0
