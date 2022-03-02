import testsrc as ai
import numpy as np

model = ai.model(1, 1, [
  ai.nn(16, ai.relu(), 0.1),
  ai.nn(16, ai.relu(), 0.1),
  ai.nn(1, ai.sigmoid(), 0.1),
])

input = np.array([0.14, 0.74, 0.58, 0.29, 0.69, 0.33])
output = np.array([0.20, 0.67, 0.78, 0.28, 0.66, 0.94])

testin = np.array([x for x in range(0, 100)]) / 100
model.train((input, output), 100)
print(model.use(testin))