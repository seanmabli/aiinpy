import numpy as np
x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
y = np.vstack(([np.hstack(([x]*2))]*2))
print(y)