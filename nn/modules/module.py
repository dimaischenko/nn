class Module(object):
    """Abstract base class for all nn modules.

    All other layers (modules) should be inherited from this class.
    """

    def __init__(self):
        pass

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *input):
        raise NotImplementedError
