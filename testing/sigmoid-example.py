import aiinpy as ai
import numpy as np

x = 0
print(ai.sigmoid().forward(x))

x = 1.5
print(ai.sigmoid().forward(x))

x = -1.5
print(ai.sigmoid().forward(x))

x = np.array([1.5, -1.5])
print(ai.sigmoid().forward(x))

sigmoid = ai.sigmoid()

x = 0
print(sigmoid.forward(x))

x = 1.5
print(sigmoid.forward(x))

x = -1.5
print(sigmoid.forward(x))

x = np.array([1.5, -1.5])
print(sigmoid.forward(x))