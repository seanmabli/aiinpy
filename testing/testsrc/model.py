import numpy as np
import sys

class model:
  def __init__(self, inshape, outshape, model):
    self.inshape = inshape = inshape if isinstance(inshape, tuple) else tuple([inshape])
    self.outshape = outshape if isinstance(outshape, tuple) else tuple([outshape])
    self.model = model

    for i in self.model:
      inshape = i.modelinit(inshape)

  def forward(self, input):
    for i in range(len(self.model)):
      input = self.model[i].forward(input)
    return input

  def backward(self, outError):
    for i in reversed(range(len(self.model))):
      outError = self.model[i].backward(outError)
    return outError

  def train(self, data, numofgen):
    # data preprocessing: tuple of (indata, outdata) with indata and outdata as numpy array
    data = list(data) if type(data) == tuple else data
    NumOfData = (set(self.inshape) ^ set(data[0].shape)).pop()
    if data[0].shape.index(NumOfData) != 0:
      x = list(range(0, len(data[0].shape)))
      x.pop(data[0].shape.index(NumOfData))
      data[0] = np.transpose(data[0], tuple([data[0].shape.index(NumOfData)]) + tuple(x))
    if data[1].shape.index(NumOfData) != 0:
      x = list(range(0, len(data[1].shape)))
      x.pop(data[1].shape.index(NumOfData))
      data[1] = np.transpose(data[1], tuple([data[1].shape.index(NumOfData)]) + tuple(x))

    # Training
    error = []
    for gen in range(numofgen):
      Random = np.random.randint(0, NumOfData)
      input = data[0][Random]
      for i in range(len(self.model)):
        input = self.model[i].forward(input)

      outError = data[1][Random] - input
      error.append(np.sum(abs(outError)))
      for i in reversed(range(len(self.model))):
        outError = self.model[i].backward(outError)

      sys.stdout.write('\r training: ' + str(gen) + '/' + str(numofgen))
      sys.stdout.flush()
    
    return error

  def test(self, data):
    # data preprocessing: tuple of (indata, outdata) with indata and outdata as numpy array
    data = list(data) if type(data) == tuple else data
    NumOfData = (set(self.inshape) ^ set(data[0].shape)).pop()
    if data[0].shape.index(NumOfData) != 0:
      x = list(range(0, len(data[0].shape)))
      x.pop(data[0].shape.index(NumOfData))
      data[0] = np.transpose(data[0], tuple([data[0].shape.index(NumOfData)]) + tuple(x))
    if data[1].shape.index(NumOfData) != 0:
      x = list(range(0, len(data[1].shape)))
      x.pop(data[1].shape.index(NumOfData))
      data[1] = np.transpose(data[1], tuple([data[1].shape.index(NumOfData)]) + tuple(x))

    # Testing
    testcorrect = 0
    for gen in range (NumOfData):
      input = data[0][gen]
      for i in range(len(self.model)):
        input = self.model[i].forward(input)

      testcorrect += 1 if np.argmax(input) == np.argmax(data[1][gen]) else 0

      sys.stdout.write('\r training: ' + str(gen) + '/' + str(NumOfData))
      sys.stdout.flush()

    return testcorrect / NumOfData