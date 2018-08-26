class Module(object):
    """Abstract base class for all nn modules.

    All other layers (modules) should be inherited from this class.
    """

    def __init__(self):
        self.output = None
        self.grad_input = None

        self.training = True

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *input):
        raise NotImplementedError
