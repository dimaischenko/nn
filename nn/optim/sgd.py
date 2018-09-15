from nn.optim import Optimizer


class SGD(Optimizer):
    """Simple SGD optimizer"""

    def __init__(self, params, grad_params, lr):
        super(SGD, self).__init__(params, grad_params)
        self.lr = lr

    def step(self):
        for param, grad in zip(self.params, self.grad_params):
            param -= self.lr * grad
