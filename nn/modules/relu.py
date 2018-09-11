from nn.modules import Module
import numpy as np


class ReLU(Module):
    """Classic ReLU non-linear layer"""

    def __init__(self):
        super(ReLU, self).__init__()
        self.retained_output = None

    def forward(self, input_):
        self.retained_output = np.maximum(0, input_)
        return self.retained_output

    def backward(self, grad_output):
        return grad_output * (self.retained_output > 0)
