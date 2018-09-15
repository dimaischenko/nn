class Module(object):
    """Abstract base class for all nn modules.

    All other layers (modules) should be inherited from this class.
    """

    def __init__(self):
        self.output = None
        self.grad_input = None
        self.params = dict()
        self.grad_params = dict()

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *input):
        raise NotImplementedError

    def get_params(self):
        """Return dict with layer params"""
        return self.params

    def get_grad_params(self):
        """Return dict with all layer tuned parameters or emtpy dict"""
        return self.grad_params

    def zero_grad(self):
        """Set gradient by all layer paramters to zero"""
        pass
