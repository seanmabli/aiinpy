import aiinpy as ai
import numpy as np

class model:
  def __init__(self, InShape, OutShape, Model):
    self.InShape, self.OutShape, self.Model = InShape, OutShape, Model

  def Train(self, TrainData, NumOfGenerations):
    NumOfData = (set(self.InShape) ^ set(TrainData.shape)).pop()
    for Generation in range (NumOfGenerations):
      Random = np.random.randint(0, NumOfData)
      In = np.take(TrainData, indices=Random, axis=int(TrainData.shape.index(NumOfData)))
      for i in range(len(self.Model)):
        print(self.Model[0].ForwardProp(1))

model = model(1, 1, [
  ai.NN(1, 1, 'Tanh', 0.01),
  ai.NN(1, 1, 'Tanh', 0.01),
  ai.NN(1, 1, 'Tanh', 0.01)
])