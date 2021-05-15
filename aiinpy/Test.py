from ActivationFunctions import LeakyReLU, DerivativeOfLeakyReLU
import torch.tensor
import torch.nn as nn
import numpy as np

m = nn.LeakyReLU(0.01)
input = torch.randn(2)
print(input)
print(LeakyReLU(np.array(input)))
print(m(input))