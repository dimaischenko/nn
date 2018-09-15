class Loss(object):
    """Base abstract class for different losses"""

    def __init__(self):
        pass

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *input):
        raise NotImplementedError
