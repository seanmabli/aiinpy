from ActivationFunctions import Sigmoid, DerivativeOfSigmoid, StableSoftMax, DerivativeOfStableSoftMax, ReLU, DerivativeOfReLU, Tanh, DerivativeOfTanh
import numpy as np

x = np.array([[0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]])
y = DerivativeOfReLU(x)
print(y)