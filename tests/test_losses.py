import unittest
import numpy as np

from nn.loss import MSE


class TestLosses(unittest.TestCase):

    def test_mse(self):
        mse = MSE()

        self.assertAlmostEqual(
            mse(np.array([[1], [0], [-1]]), np.array([1, 0, -1])),
            0.0
        )

        self.assertAlmostEqual(
            mse(np.array([[1], [0], [-1]]), np.array([0, 0, 0])),
            2.0 / 3
        )

        self.assertAlmostEqual(
            mse(np.array([[1.0], [0], [-1]]), np.array([0, 1, 0])),
            1.0
        )

        np.testing.assert_array_almost_equal(
            mse.backward(np.array([[1.0], [0], [-1]]), np.array([0, 1, 0])),
            np.array([[2. / 3 * 1], [-2. / 3 * 1], [-2. / 3 * 1]])
        )

        np.testing.assert_array_almost_equal(
            mse.backward(np.array([[0], [0], [0]]), np.array([0, 0, 0])),
            np.array([[0], [0], [0]])
        )
