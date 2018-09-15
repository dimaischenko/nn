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
        self.params = {
            "W": 0.01 * np.random.rand(self.input_size, self.output_size),
            "b": np.zeros((1, self.output_size))
        }

        self.grad_params = {"W": 0, "b": 0}

    def forward(self, input_):
        self.output = np.dot(input_, self.params["W"]) + self.params["b"]
        return self.output

    def backward(self, grad_output):
        self.grad_input = np.dot(grad_output, (self.params["W"]).T)
        return self.grad_input

    def zero_grad(self):
        self.grad_params["W"] = 0
        self.grad_params["b"] = 0
