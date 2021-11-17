import numpy as np
import testsrc as ai

PopulationSize = 1000
Board = 8

model = ai.neuroevolution((8, 8), 4, PopulationSize, [
  ai.conv((8, 8), (16, 3, 3), 0.01, ai.sigmoid, True),
  ai.pool((2, 2), (2, 2), 'Max'),
  ai.nn((16, 4, 4), 4, ai.stablesoftmax, 0.1)
])

Alive = np.array([True] * PopulationSize, dtype=bool)
Score = np.zeros(PopulationSize)

Snake = np.zeros((PopulationSize, 2, 1), dtype=int)
Apple = np.zeros((2, PopulationSize), dtype=int)
for i in range(Apple.shape[1]):
  while np.sum(Apple[:, i]) == 0:
    Apple[:, i] = np.random.randint(Board, size=2)

for Generation in range(100):
  while np.sum(Alive) != 0:
    for x in range(PopulationSize):
      print(Snake.shape)
      In = np.zeros((Board, Board))
      for i in range(Snake.shape[0]):
        In[Snake[x, i, 0], Snake[x, i, 1]] = 1
      In[Apple[x, 0], Apple[x, 1]] = 0.5
      Out = model.forwardsingle(In, x)
      if (np.max(Out) == Out[0]):
        Snake[:, :, x] = np.vstack(([Snake[0, 0, x], Snake[0, 1, x] - 1], Snake[:-1, :, x]))
      if (np.max(Out) == Out[1]):
        Snake[:, :, x] = np.vstack(([Snake[0, 0, x] + 1, Snake[0, 1, x]], Snake[:-1, :, x]))
      if (np.max(Out) == Out[2]):
        Snake[:, :, x] = np.vstack(([Snake[0, 0, x], Snake[0, 1, x] + 1], Snake[:-1, :, x]))
      if (np.max(Out) == Out[3]):
        Snake[:, :, x] = np.vstack(([Snake[0, 0, x] - 1, Snake[0, 1, x]], Snake[:-1, :, x]))

      # Increase score
      if np.array_equal(Snake[x, 0, :], Apple[x, :]) and Alive[x] == True:
        Score[x] += 1
        Snake[x, :, :] = np.vstack((Snake[x, :, :], [Board, Board]))

      # Check that new apple location is not on the snake
      for i in range(Snake.shape[0]):
        if np.array_equal(Apple[0], Snake[i, 0, x]) and np.array_equal(Apple[1], Snake[i, 1, x]):
          Apple = np.random.randint(Board, size=2)
          i = 0
          
      # Check out of bounds
      if Snake[0, 0] < 0 or Snake[0, 0] > Board - 1 or Snake[0, 1] < 0 or Snake[0, 1] > Board - 1:
        done = True
      
      # Check intersecting with itself
      for i in range(Snake.shape[0] - 1):
        for j in range(Snake.shape[0] - 1):
          if i != j and np.array_equal(Snake[i, :], Snake[j, :]):
            Alive[x] = False