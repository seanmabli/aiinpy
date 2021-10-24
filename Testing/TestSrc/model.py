import numpy as np
from alive_progress import alive_bar
import sys

class model:
  def __init__(self, InShape, OutShape, Model):
    self.InShape, self.OutShape, self.Model = tuple([InShape]), tuple([OutShape]), Model

  def train(self, InTrainData, OutTrainData, NumOfGenerations):
    NumOfData = (set(self.InShape) ^ set(InTrainData.shape)).pop()
    with alive_bar(NumOfGenerations) as bar:
      for Generation in range (NumOfGenerations):
        In = np.take(InTrainData, indices=Generation, axis=int(InTrainData.shape.index(NumOfData)))
        for i in range(len(self.Model)):
          In = self.Model[i].forwardprop(In)

        OutError = np.take(OutTrainData, indices=Generation, axis=int(OutTrainData.shape.index(NumOfData))) - In
        for i in reversed(range(len(self.Model))):
          OutError = self.Model[i].backprop(OutError)
        bar()

  def test(self, InTestData, OutTestData):
    NumOfData = (set(self.InShape) ^ set(InTestData.shape)).pop()
    testcorrect = 0
    with alive_bar(NumOfData) as bar:
      for Generation in range (NumOfData):
        In = np.take(InTestData, indices=Generation, axis=int(InTestData.shape.index(NumOfData)))
        
        for i in range(len(self.Model)):
          In = self.Model[i].forwardprop(In)

        testcorrect += 1 if np.argmax(In) == np.argmax(np.take(OutTestData, indices=Generation, axis=int(OutTestData.shape.index(NumOfData)))) else 0
        bar()
    return testcorrect / NumOfData