from nn.modules import Module
import numpy as np


class ReLU(Module):
    """Classic ReLU non-linear layer"""

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input_):
        self.output = np.maximum(0, input_)
        return self.output

    def backward(self, grad_output):
        self.grad_input = grad_output * (self.output > 0)
        return self.grad_input
