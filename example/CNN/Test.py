import numpy as np

x = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
y = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
z = np.equal(x, y).astype(int)
print(z)