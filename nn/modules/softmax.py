from nn.modules import Module


class LogSoftMax(Module):
    """Logarithmic softmax layer"""

    def __init__(self):
        super(LogSoftMax, self).__init__()

    def forward(self, input_):
        pass

    def backward(self, grad_output):
        pass
