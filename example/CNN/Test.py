import numpy as np

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

a = np.zeros((3, 3))

for i in range(3):
  for j in range(3):
    a[i, j] = np.sum(np.multiply(x, np.pad(y, 1)[i:i+2, j:j+2]))
    
print(a)