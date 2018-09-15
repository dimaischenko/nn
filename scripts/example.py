import numpy as np

from nn.modules import Linear
from nn.modules import ReLU
from nn.modules import Sequential
from nn.optim import SGD
from nn.loss import MSE


network = Sequential([
    Linear(4, 3),
    ReLU(),
    Linear(3, 2),
    ReLU(),
    Linear(2, 1)
])

print("Params")
print(network.get_params())
print(network.get_grad_params())

network.zero_grad()

X = np.array([
    [1, 2.5, 3, 4.1],
    [-1.2, 3, 0.2, 0]
])

Y = np.array([1, 0])

optim = SGD(
    network.get_params(),
    network.get_grad_params(),
    lr=0.01
)

optim.step()

criterion = MSE()
loss = criterion(network.forward(X), Y)
print("Loss", loss)
print("Loss-backward", criterion.backward(network.forward(X), Y))


print("Outputs")
print(network.forward(X))
print(network.backward([[1], [1]]))
