import src as ai
import numpy as np
import time

starttime = time.time()
activation = ai.mish()
x = np.random.rand(1000000)
y = activation.forward(x)
der = activation.backward(y)
print(time.time() - starttime)