import numpy as np

x = np.random.binomial(1, 0.5, size=10)
x = np.where(x == 0, 1, 0)
print(x)