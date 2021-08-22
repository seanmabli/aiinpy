import numpy as np

a = np.zeros((32, 32))

for i in range(14):
  for j in range(14):
    a[i * 2 : i * 2 + 3, j * 2 : j * 2 + 3] += np.ones((3, 3))

print(a[0 : 16, 0 : 16])
print(a[0 : 16, 16 : 32])
print(a[16 : 32, 0 : 16])
print(a[16 : 32, 16 : 32])