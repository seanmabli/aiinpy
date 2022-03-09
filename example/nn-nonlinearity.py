import src as ai
import numpy as np
import matplotlib.pyplot as plt

model = ai.model(1, 1, [
  ai.nn(32, ai.relu(), 0.01),
  ai.nn(32, ai.relu(), 0.01),
  ai.nn(1, ai.sigmoid(), 0.01),
])

def f(x):
  return np.log(x * 100 + 1) / 5

input = np.arange(50) / 100
notgiven = np.vectorize(f)(np.arange(50, 100) / 100)
output = np.vectorize(f)(input)

model.train((input, output), 1000000)

testin = np.array([x for x in range(0, 100)]) / 100
line = model.use(testin)

for i in range(len(line)):
  plt.scatter(testin[i], line[i], color=(1, 0, 0), s=1)

for i in range(len(input)):
  plt.scatter(input[i], output[i], color=(0, 0, 1), s=1)

for i in range(len(notgiven)):
  plt.scatter((np.arange(50, 100) / 100)[i], notgiven[i], color=(0, 1, 0), s=1)

plt.show()