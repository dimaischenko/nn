import unittest
import numpy as np

from nn.modules import Linear
from nn.modules import Sequential


class TestModules(unittest.TestCase):

    def test_linear(self):
        layer = Linear(10, 5)
        x = np.zeros((2, 10))

        output = layer(x)
        np.testing.assert_array_almost_equal(output, np.zeros((2, 5)))

    def test_sequential(self):
        seq = Sequential()

        seq.add(Linear(10, 5))
        seq.add(Linear(5, 2))

        x = np.zeros((2, 10))
        output = seq(x)
        np.testing.assert_array_almost_equal(output, np.zeros((2, 2)))

        grad_input = seq.backward(np.zeros((2, 2)))
        np.testing.assert_array_almost_equal(grad_input, np.zeros((2, 10)))
