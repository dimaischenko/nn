class Module(object):
    """Abstract base class for all nn modules.

    All other layers (modules) should be inherited from this class.
    """

    def __init__(self):
        self.output = None
        self.grad_input = None
        self.params = []
        self.grad_params = []

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *input):
        raise NotImplementedError

    def get_params(self):
        """Return list with layer params"""
        return self.params

    def get_grad_params(self):
        """Return list with all layer tuned parameters or emtpy list"""
        return self.grad_params

    def zero_grad(self):
        """Set gradient by all layer paramters to zero"""
        pass
