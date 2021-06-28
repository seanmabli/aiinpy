import numpy as np

a = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

Stride = (2, 2)

if Stride != (1, 1):
  a = np.insert(np.insert(a, np.arange(Stride[0] - 1, a.shape[0] + 1, Stride[0] - 1), 0, axis=0), np.arange(Stride[0] - 1, a.shape[0], Stride[1] - 1), 0, axis=1)
