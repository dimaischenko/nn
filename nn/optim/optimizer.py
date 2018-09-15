class Optimizer(object):
    """Base abstract class for all optimizers

    Get network parameters and its gradients and
    create steps
    """

    def __init__(self, params, grad_params):
        self.params = params
        self.grad_params = grad_params

    def step(self):
        for param, grad in zip(self.params, self.grad_params):
            param += self.lr * grad
