from ActivationFunctions import DerivativeOfStableSoftMax, DerivativeOfStableSoftMaxTest, EquationForDerivativeOfStableSoftMax
import numpy as np

'''
import torch.tensor
import torch.nn as nn

m = nn.LeakyReLU(0.01)
input = torch.randn(2)
print(input)
print(LeakyReLU(np.array(input)))
print(m(input))
'''

x = np.array([[0.13, 0.27, 0.1], [0.13, 0.27, 0.1]])
print(DerivativeOfStableSoftMaxTest(x))
print(DerivativeOfStableSoftMax(x))