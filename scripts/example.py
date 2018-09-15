import numpy as np

from nn.modules import Linear
from nn.modules import ReLU
from nn.modules import Sequential


network = Sequential([
    Linear(4, 3),
    ReLU(),
    Linear(3, 2),
    ReLU(),
    Linear(2, 1)
])

X = np.array([
    [1, 2.5, 3, 4.1],
    [-1.2, 3, 0.2, 0]
])


print(network.forward(X))
print(network.backward([[1], [1]]))
