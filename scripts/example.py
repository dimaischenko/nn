import numpy as np

from nn.modules import Linear
from nn.modules import ReLU

linear = Linear(10, 20)
relu = ReLU()

print(relu.forward(np.array([-1, 2])))
print(relu.backward(np.array(1)))
