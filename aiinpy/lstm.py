import numpy as np
from .binarystep import binarystep
from .gaussian import gaussian
from .identity import identity
from .leakyrelu import leakyrelu
from .mish import mish
from .relu import relu
from .selu import selu
from .sigmoid import sigmoid
from .silu import silu
from .softmax import softmax
from .softplus import softplus
from .stablesoftmax import stablesoftmax
from .tanh import tanh

class lstm:
  def __init__(self, outshape, outactivation, hidshape=64, learningrate=0.05, inshape=None):
    self.outactivation, self.learningrate = outactivation, learningrate
    self.inshape, self.hidshape, self.outshape = inshape, hidshape, outshape
    
    if inshape is not None:
      self.weightsinToForgetGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
      self.weightsinToinGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
      self.weightsinTooutGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
      self.weightsinTocellMemGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))

    self.weightshidToForgetGate = np.random.uniform(-0.005, 0.005, (hidshape, hidshape))
    self.weightshidToinputGate = np.random.uniform(-0.005, 0.005, (hidshape, hidshape))
    self.weightshidTooutputGate = np.random.uniform(-0.005, 0.005, (hidshape, hidshape))
    self.weightshidTocellMemGate = np.random.uniform(-0.005, 0.005, (hidshape, hidshape))

    self.ForgetGatebiases = np.zeros(hidshape)
    self.inGatebiases = np.zeros(hidshape)
    self.outGatebiases = np.zeros(hidshape)
    self.cellMemGatebiases = np.zeros(hidshape)

    self.weightshidToout = np.random.uniform(-0.005, 0.005, (hidshape, outshape))
    self.outbias = np.zeros(outshape)

  def __copy__(self):
    return type(self)(self.outshape, self.outactivation, self.hidshape, self.learningrate, self.inshape)

  def modelinit(self, inshape):
    self.inshape = inshape
    self.weightsinToForgetGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
    self.weightsinToinGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
    self.weightsinTooutGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
    self.weightsinTocellMemGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
    return self.outshape

  def forward(self, input):
    self.input = input
    self.cellSize = len(input)

    self.hid = np.zeros((self.cellSize + 1, self.hidshape))
    self.cellMem = np.zeros((self.cellSize + 1, self.hidshape))
    self.out = np.zeros((self.cellSize, self.outshape))

    self.ForgetGate = np.zeros((self.cellSize, self.hidshape))
    self.inGate = np.zeros((self.cellSize, self.hidshape))
    self.outGate = np.zeros((self.cellSize, self.hidshape))
    self.cellMemGate = np.zeros((self.cellSize, self.hidshape))

    for i in range(self.cellSize):
      self.ForgetGate[i, :] = sigmoid().forward((self.weightsinToForgetGate.T @ self.input[i, :]) + (self.weightshidToForgetGate.T @ self.hid[i, :]) + self.ForgetGatebiases)
      self.inGate[i, :] = sigmoid().forward((self.weightsinToinGate.T @ self.input[i, :]) + (self.weightshidToinputGate.T @ self.hid[i, :]) + self.inGatebiases)
      self.outGate[i, :] = sigmoid().forward((self.weightsinTooutGate.T @ self.input[i, :]) + (self.weightshidTooutputGate.T @ self.hid[i, :]) + self.outGatebiases)
      self.cellMemGate[i, :] = tanh().forward((self.weightsinTocellMemGate.T @ self.input[i, :]) + (self.weightshidTocellMemGate.T @ self.hid[i, :]) + self.cellMemGatebiases)

      self.cellMem[i + 1, :] = (self.ForgetGate[i, :] * self.cellMem[i, :]) + (self.inGate[i, :] * self.cellMemGate[i, :])
      self.hid[i + 1, :] = self.outGate[i, :] * tanh().forward(self.cellMem[i + 1, :])
      self.out[i, :] = self.outactivation.forward(self.weightshidToout.T @ self.hid[i + 1, :] + self.outbias)

    return self.out

  def backward(self, outError):
    inError = np.zeros(self.input.shape)
    hidError = np.zeros(self.hidshape)
    cellMemError = np.zeros(self.hidshape)

    weightsinToForgetGate?? = np.zeros(self.weightsinToForgetGate.shape)
    weightsinToinGate?? = np.zeros(self.weightsinToinGate.shape)
    weightsinTooutGate?? = np.zeros(self.weightsinTooutGate.shape)
    weightsinTocellMemGate?? = np.zeros(self.weightsinTocellMemGate.shape)

    weightshidToForgetGate?? = np.zeros(self.weightshidToForgetGate.shape)
    weightshidToinputGate?? = np.zeros(self.weightshidToinputGate.shape)
    weightshidTooutputGate?? = np.zeros(self.weightshidTooutputGate.shape)
    weightshidTocellMemGate?? = np.zeros(self.weightshidTocellMemGate.shape)

    ForgetGatebiases?? = np.zeros(self.ForgetGatebiases.shape)
    inGatebiases?? = np.zeros(self.inGatebiases.shape)
    outGatebiases?? = np.zeros(self.outGatebiases.shape)
    cellMemGatebiases?? = np.zeros(self.cellMemGatebiases.shape)

    weightshidToout?? = np.zeros(self.weightshidToout.shape)
    outbias?? = np.zeros(self.outbias.shape)

    for i in reversed(range(self.cellSize)):
      outGradient = self.outactivation.backward(self.out[i, :]) * outError[i, :]
      
      hidError += self.weightshidToout @ outError[i, :]

      cellMemError += hidError * self.outGate[i] * tanh().backward(tanh().forward(self.cellMem[i + 1, :]))
      outGateError = hidError * tanh().forward(self.cellMem[i + 1, :])

      ForgetGateError = cellMemError * self.cellMem[i, :]
      inGateError = cellMemError * self.cellMemGate[i, :]
      cellMemGateError = cellMemError * self.inGate[i, :]

      cellMemError *= self.ForgetGate[i, :]

      ForgetGateGradient = sigmoid().backward(self.ForgetGate[i, :]) * ForgetGateError
      inGateGradient = sigmoid().backward(self.inGate[i, :]) * inGateError
      outGateGradient = sigmoid().backward(self.outGate[i, :]) * outGateError
      cellMemGateGradient = tanh().backward(self.cellMemGate[i, :]) * cellMemGateError

      hidError = self.weightshidToForgetGate @ ForgetGateError + self.weightshidToinputGate @ inGateError + self.weightshidTooutputGate @ outGateError + self.weightshidTocellMemGate @ cellMemGateError
      inError[i, :] = self.weightsinToForgetGate @ ForgetGateError + self.weightsinToinGate @ inGateError + self.weightsinTooutGate @ outGateError + self.weightsinTocellMemGate @ cellMemGateError

      weightsinToForgetGate?? += np.outer(self.input[i, :].T, ForgetGateGradient)
      weightsinToinGate?? += np.outer(self.input[i, :].T, inGateGradient)
      weightsinTooutGate?? += np.outer(self.input[i, :].T, outGateGradient)
      weightsinTocellMemGate?? += np.outer(self.input[i, :].T, cellMemGateGradient)

      weightshidToForgetGate?? += np.outer(self.hid[i, :].T, ForgetGateGradient)
      weightshidToinputGate?? += np.outer(self.hid[i, :].T, inGateGradient)
      weightshidTooutputGate?? += np.outer(self.hid[i, :].T, outGateGradient)
      weightshidTocellMemGate?? += np.outer(self.hid[i, :].T, cellMemGateGradient)

      ForgetGatebiases?? += ForgetGateGradient
      inGatebiases?? += inGateGradient
      outGatebiases?? += outGateGradient
      cellMemGatebiases?? += cellMemGateGradient

      weightshidToout?? += np.outer(self.hid[i].T, outGradient)
      outbias?? += outGradient

    self.weightsinToForgetGate += np.clip(weightsinToForgetGate??, -1, 1) * self.learningrate
    self.weightsinToinGate += np.clip(weightsinToinGate??, -1, 1) * self.learningrate
    self.weightsinTooutGate += np.clip(weightsinTooutGate??, -1, 1) * self.learningrate
    self.weightsinTocellMemGate += np.clip(weightsinTocellMemGate??, -1, 1) * self.learningrate

    self.weightshidToForgetGate += np.clip(weightshidToForgetGate??, -1, 1) * self.learningrate
    self.weightshidToinputGate += np.clip(weightshidToinputGate??, -1, 1) * self.learningrate
    self.weightshidTooutputGate += np.clip(weightshidTooutputGate??, -1, 1) * self.learningrate
    self.weightshidTocellMemGate += np.clip(weightshidTocellMemGate??, -1, 1) * self.learningrate

    self.ForgetGatebiases += np.clip(ForgetGatebiases??, -1, 1) * self.learningrate
    self.inGatebiases += np.clip(inGatebiases??, -1, 1) * self.learningrate
    self.outGatebiases += np.clip(outGatebiases??, -1, 1) * self.learningrate
    self.cellMemGatebiases += np.clip(cellMemGatebiases??, -1, 1) * self.learningrate

    self.weightshidToout += np.clip(weightshidToout??, -1, 1) * self.learningrate
    self.outbias += np.clip(outbias??, -1, 1) * self.learningrate

    return inError