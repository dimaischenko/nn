class Layer(object):
    """Abstract base class for layer.

    All other layers should be inherited from this class.
    """

    def __init__(self):
        pass

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *input):
        raise NotImplementedError
