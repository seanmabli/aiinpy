import src as ai
import numpy as np

a = ai.sigmoid()
x = np.random.rand(5)
y = a.forward(x)
z = a.backward(y)
print(y)
print(z)