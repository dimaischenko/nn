class Loss(object):
    """Base abstract class for different losses"""

    def __init__(self):
        pass

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input):
        return self.forward(*input)

    def backward(self, *input):
        raise NotImplementedError
