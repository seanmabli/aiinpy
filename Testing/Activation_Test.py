import numpy as np
from TestSrc.Activation import *

x = np.random.uniform(-1, 1, (10))

y = ActivationDerivative(x, 'LeakyReLU')

print(y)
