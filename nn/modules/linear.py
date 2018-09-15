from nn.modules import Module
import numpy as np


class Linear(Module):
    """Simple fully connected layer"""

    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.last_input = None

        self._init_params()

    def _init_params(self):
        self.W = 0.01 * np.random.rand(self.input_size, self.output_size)
        self.b = np.zeros((1, self.output_size))

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def get_params(self):
        return [self.W, self.b]

    def get_grad_params(self):
        return [self.dW, self.db]

    def forward(self, input_):
        self.last_input = input_
        self.output = np.dot(input_, self.W) + self.b

        return self.output

    def backward(self, grad_output):
        output_size = grad_output.shape[0]

        self.dW[:, :] = np.dot((self.last_input).T, grad_output)
        self.db[:, :] = np.dot(np.ones((1, output_size)), grad_output)

        self.grad_input = np.dot(grad_output, (self.W).T)

        return self.grad_input

    def zero_grad(self):
        self.dW.fill(0)
        self.db.fill(0)
