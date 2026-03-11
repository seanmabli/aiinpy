import numpy as np
import sys

class model:
  def __init__(self, inshape, outshape, model):
    self.inshape = inshape = inshape if isinstance(inshape, tuple) else (inshape,)
    self.outshape = outshape if isinstance(outshape, tuple) else (outshape,)
    self.model = model

    for layer in self.model:
      inshape = layer.modelinit(inshape)

  def forward(self, input):
    for layer in self.model:
      input = layer.forward(input)
    return input

  def backward(self, outError):
    for layer in reversed(self.model):
      outError = layer.backward(outError)
    return outError

  def train(self, data, numofgen):
    data = list(data) if type(data) == tuple else data
    data[0] = np.reshape(data[0], (data[0].shape[0], 1)) if len(data[0].shape) == 1 else data[0]
    data[1] = np.reshape(data[1], (data[1].shape[0], 1)) if len(data[1].shape) == 1 else data[1]

    NumOfData = (set(self.inshape) ^ set(data[0].shape)).pop()
    if data[0].shape.index(NumOfData) != 0:
      x = list(range(0, len(data[0].shape)))
      x.pop(data[0].shape.index(NumOfData))
      data[0] = np.transpose(data[0], (data[0].shape.index(NumOfData),) + tuple(x))
    if data[1].shape.index(NumOfData) != 0:
      x = list(range(0, len(data[1].shape)))
      x.pop(data[1].shape.index(NumOfData))
      data[1] = np.transpose(data[1], (data[1].shape.index(NumOfData),) + tuple(x))

    error = []
    print('')
    for gen in range(numofgen):
      random = np.random.randint(0, NumOfData)
      output = self.forward(data[0][random])

      outError = data[1][random] - output
      error.append(np.sum(np.abs(outError)))
      self.backward(outError)
      sys.stdout.write('\r training: ' + str(gen + 1) + '/' + str(numofgen))
    
    return error

  def test(self, data):
    data = list(data) if type(data) == tuple else data
    data[0] = np.reshape(data[0], (data[0].shape[0], 1)) if len(data[0].shape) == 1 else data[0]
    data[1] = np.reshape(data[1], (data[1].shape[0], 1)) if len(data[1].shape) == 1 else data[1]

    NumOfData = (set(self.inshape) ^ set(data[0].shape)).pop()
    if data[0].shape.index(NumOfData) != 0:
      x = list(range(0, len(data[0].shape)))
      x.pop(data[0].shape.index(NumOfData))
      data[0] = np.transpose(data[0], (data[0].shape.index(NumOfData),) + tuple(x))
    if data[1].shape.index(NumOfData) != 0:
      x = list(range(0, len(data[1].shape)))
      x.pop(data[1].shape.index(NumOfData))
      data[1] = np.transpose(data[1], (data[1].shape.index(NumOfData),) + tuple(x))

    testcorrect = 0
    print('')
    for gen in range(NumOfData):
      output = self.forward(data[0][gen])
      testcorrect += 1 if np.argmax(output) == np.argmax(data[1][gen]) else 0
      
    return testcorrect / NumOfData

  def use(self, indata):
    indata = np.reshape(indata, (indata.shape[0], 1)) if len(indata.shape) == 1 else indata
    NumOfData = (set(self.inshape) ^ set(indata.shape)).pop()
    if indata.shape.index(NumOfData) != 0:
      x = list(range(0, len(indata.shape)))
      x.pop(indata.shape.index(NumOfData))
      indata = np.transpose(indata, (indata.shape.index(NumOfData),) + tuple(x))

    outdata = np.zeros(indata.shape)
    for gen in range(NumOfData):
      outdata[gen] = self.forward(indata[gen])
    return outdata
