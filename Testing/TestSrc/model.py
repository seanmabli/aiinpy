import numpy as np
from alive_progress import alive_bar
import sys

class model:
  def __init__(self, InShape, OutShape, Model):
    self.InShape, self.OutShape, self.Model = InShape if isinstance(InShape, tuple) else tuple([InShape]),  OutShape if isinstance(OutShape, tuple) else tuple([OutShape]), Model

  def train(self, InTrainData, OutTrainData):
    # Data Preprocessing
    NumOfData = (set(self.InShape) ^ set(InTrainData.shape)).pop()
    if InTrainData.shape.index(NumOfData) != 0:
      x = list(range(0, len(InTrainData.shape)))
      x.pop(InTrainData.shape.index(NumOfData))
      InTrainData = np.transpose(InTrainData, tuple([InTrainData.shape.index(NumOfData)]) + tuple(x))
    if OutTrainData.shape.index(NumOfData) != 0:
      x = list(range(0, len(OutTrainData.shape)))
      x.pop(OutTrainData.shape.index(NumOfData))
      OutTrainData = np.transpose(OutTrainData, tuple([OutTrainData.shape.index(NumOfData)]) + tuple(x))

    # Training
    with alive_bar(NumOfData) as bar:
      for Generation in range (NumOfData):
        In = InTrainData[Generation]
        for i in range(len(self.Model)):
          In = self.Model[i].forwardprop(In)

        OutError = OutTrainData[Generation] - In
        for i in reversed(range(len(self.Model))):
          OutError = self.Model[i].backprop(OutError)

        bar()

  def test(self, InTestData, OutTestData):
    # Data Preprocessing
    NumOfData = (set(self.InShape) ^ set(InTestData.shape)).pop()
    if InTestData.shape.index(NumOfData) != 0:
      x = list(range(0, len(InTestData.shape)))
      x.pop(InTestData.shape.index(NumOfData))
      InTestData = np.transpose(InTestData, tuple([InTestData.shape.index(NumOfData)]) + tuple(x))
    if OutTestData.shape.index(NumOfData) != 0:
      x = list(range(0, len(OutTestData.shape)))
      x.pop(OutTestData.shape.index(NumOfData))
      OutTestData = np.transpose(OutTestData, tuple([OutTestData.shape.index(NumOfData)]) + tuple(x))

    # Testing
    testcorrect = 0
    with alive_bar(NumOfData) as bar:
      for Generation in range (NumOfData):
        In = InTestData[Generation]
        for i in range(len(self.Model)):
          In = self.Model[i].forwardprop(In)

        testcorrect += 1 if np.argmax(In) == np.argmax(OutTestData[Generation]) else 0
        bar()

    return testcorrect / NumOfData