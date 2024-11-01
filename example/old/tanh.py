import numpy as np

class tanh:
  def __repr__(self):
    return 'tanh()'

  # def forwardone(self, input):
  #   return (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))

  # def forwardtwo(self, input):
  #   return (np.exp(2 * input) - 1) / (np.exp(2 * input) + 1)

  # def forwardthree(self, input):
  #   a = np.exp(input)
  #   return (a - 1 / a) / (a + 1 / a)

  def forward(self, input):
    a = np.exp(2 * input)
    return (a - 1) / (a + 1)

  # def backwardone(self, input):
  #   return (4 * np.exp(2 * input)) / np.square(np.exp(2 * input) + 1)

  # def backwardtwo(self, input):
  #   return 4 / np.square(np.exp(input) + np.exp(-input))

  # def backwardthree(self, input):
  #   a = np.exp(2 * input)
  #   return (4 * a) / np.square(a + 1)
  
  def backward(self, input):
    a = np.exp(input)
    return 4 / np.square(a + 1 / a)

# forward
# import timeit
# print(timeit.timeit('(np.exp(2 * np.random.uniform(-1, 1, (100, 100))) - 1) / (np.exp(2 * np.random.uniform(-1, 1, (100, 100))) + 1)', setup='import numpy as np', number=10000))
# print(timeit.timeit('(np.exp(np.random.uniform(-1, 1, (100, 100))) - np.exp(-np.random.uniform(-1, 1, (100, 100)))) / (np.exp(np.random.uniform(-1, 1, (100, 100))) + np.exp(-np.random.uniform(-1, 1, (100, 100))))', setup='import numpy as np', number=10000))
# print(timeit.timeit('a = np.exp(np.random.uniform(-1, 1, (100, 100))); (a - 1 / a) / (a + 1 / a)', setup='import numpy as np', number=10000))
# print(timeit.timeit('a = np.exp(2 * np.random.uniform(-1, 1, (100, 100))); (a - 1) / (a + 1)', setup='import numpy as np', number=10000))

# backward
# import timeit
# print(timeit.timeit('(4 * np.exp(2 * np.random.uniform(-1, 1, (100, 100)))) / np.square(np.exp(2 * np.random.uniform(-1, 1, (100, 100))) + 1)', setup='import numpy as np', number=10000)) # 2.7722288
# print(timeit.timeit('4 / np.square(np.exp(np.random.uniform(-1, 1, (100, 100))) + np.exp(-np.random.uniform(-1, 1, (100, 100))))', setup='import numpy as np', number=10000)) # 2.6440156000000004
# print(timeit.timeit('a = np.exp(2 * np.random.uniform(-1, 1, (100, 100))); (4 * a) / np.square(a + 1)', setup='import numpy as np', number=10000)) # 1.4674546
# print(timeit.timeit('a = np.exp(np.random.uniform(-1, 1, (100, 100))); 4 / np.square(a + 1 / a)', setup='import numpy as np', number=10000)) # 1.4442597