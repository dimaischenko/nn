from nn.loss import Loss
import numpy as np


class MSE(Loss):
    """Mean squared error loss"""

    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, input_, target):
        # TODO(dima): think about vector from (B x 1) matrix with .T[0]
        return np.mean((input_.T[0] - target) ** 2)

    def backward(self, input_, target):
        return np.array([2 / input_.shape[0] * (input_.T[0] - target)]).T
