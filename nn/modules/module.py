class Module(object):
    """Abstract base class for all nn modules.

    All other layers (modules) should be inherited from this class.
    """

    def __init__(self):
        self.output = None
        self.grad_input = None

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *input):
        raise NotImplementedError

    def get_params(self):
        """Return list with params for all layers"""
        return []

    def get_grad_params(self):
        """Return list with grad params for all layers"""
        return []

    def zero_grad(self):
        """Set gradient by all layer paramters to zero"""
        pass
