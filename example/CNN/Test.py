import numpy as np

x = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])

print(x)
x = np.flip(x, (1, 2))
print(x)