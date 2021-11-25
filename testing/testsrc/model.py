import numpy as np
from alive_progress import alive_bar

class model:
  def __init__(self, inshape, outshape, Model):
    self.inshape = inshape if isinstance(inshape, tuple) else tuple([inshape])
    self.outshape = outshape if isinstance(outshape, tuple) else tuple([outshape])
    self.Model = Model
  
  def forward(self, input):
    for i in range(len(self.Model)):
      input = self.Model[i].forward(input)
    return input

  def backward(self, outError):
    for i in reversed(range(len(self.Model))):
      outError = self.Model[i].backward(outError)
    return outError

  def train(self, inTrainData, outTrainData, NumOfGen):
    # Data Preprocessing
    NumOfData = (set(self.inshape) ^ set(inTrainData.shape)).pop()
    if inTrainData.shape.index(NumOfData) != 0:
      x = list(range(0, len(inTrainData.shape)))
      x.pop(inTrainData.shape.index(NumOfData))
      inTrainData = np.transpose(inTrainData, tuple([inTrainData.shape.index(NumOfData)]) + tuple(x))
    if outTrainData.shape.index(NumOfData) != 0:
      x = list(range(0, len(outTrainData.shape)))
      x.pop(outTrainData.shape.index(NumOfData))
      outTrainData = np.transpose(outTrainData, tuple([outTrainData.shape.index(NumOfData)]) + tuple(x))

    # Training
    with alive_bar(NumOfGen) as bar:
      for _ in range (NumOfGen):
        Random = np.random.randint(0, NumOfData)
        input = inTrainData[Random]
        for i in range(len(self.Model)):
          input = self.Model[i].forward(input)

        outError = outTrainData[Random] - input
        for i in reversed(range(len(self.Model))):
          outError = self.Model[i].backward(outError)

        bar()

  def test(self, inTestData, outTestData):
    # Data Preprocessing
    NumOfData = (set(self.inshape) ^ set(inTestData.shape)).pop()
    if inTestData.shape.index(NumOfData) != 0:
      x = list(range(0, len(inTestData.shape)))
      x.pop(inTestData.shape.index(NumOfData))
      inTestData = np.transpose(inTestData, tuple([inTestData.shape.index(NumOfData)]) + tuple(x))
    if outTestData.shape.index(NumOfData) != 0:
      x = list(range(0, len(outTestData.shape)))
      x.pop(outTestData.shape.index(NumOfData))
      outTestData = np.transpose(outTestData, tuple([outTestData.shape.index(NumOfData)]) + tuple(x))

    # Testing
    testcorrect = 0
    with alive_bar(NumOfData) as bar:
      for Generation in range (NumOfData):
        input = inTestData[Generation]
        for i in range(len(self.Model)):
          input = self.Model[i].forward(input)

        testcorrect += 1 if np.argmax(input) == np.argmax(outTestData[Generation]) else 0
        bar()

    return testcorrect / NumOfData