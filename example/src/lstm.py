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

    self.weightsHidToForgetGate = np.random.uniform(-0.005, 0.005, (hidshape, hidshape))
    self.weightsHidToinputGate = np.random.uniform(-0.005, 0.005, (hidshape, hidshape))
    self.weightsHidTooutputGate = np.random.uniform(-0.005, 0.005, (hidshape, hidshape))
    self.weightsHidTocellMemGate = np.random.uniform(-0.005, 0.005, (hidshape, hidshape))

    self.ForgetGatebiases = np.zeros(hidshape)
    self.inGatebiases = np.zeros(hidshape)
    self.outGatebiases = np.zeros(hidshape)
    self.cellMemGatebiases = np.zeros(hidshape)

    self.weightsHidToout = np.random.uniform(-0.005, 0.005, (hidshape, outshape))
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

    self.Hid = np.zeros((self.cellSize + 1, self.hidshape))
    self.cellMem = np.zeros((self.cellSize + 1, self.hidshape))
    self.out = np.zeros((self.cellSize, self.outshape))

    self.ForgetGate = np.zeros((self.cellSize, self.hidshape))
    self.inGate = np.zeros((self.cellSize, self.hidshape))
    self.outGate = np.zeros((self.cellSize, self.hidshape))
    self.cellMemGate = np.zeros((self.cellSize, self.hidshape))

    for i in range(self.cellSize):
      self.ForgetGate[i, :] = sigmoid().forward((self.weightsinToForgetGate.T @ self.input[i, :]) + (self.weightsHidToForgetGate.T @ self.Hid[i, :]) + self.ForgetGatebiases)
      self.inGate[i, :] = sigmoid().forward((self.weightsinToinGate.T @ self.input[i, :]) + (self.weightsHidToinputGate.T @ self.Hid[i, :]) + self.inGatebiases)
      self.outGate[i, :] = sigmoid().forward((self.weightsinTooutGate.T @ self.input[i, :]) + (self.weightsHidTooutputGate.T @ self.Hid[i, :]) + self.outGatebiases)
      self.cellMemGate[i, :] = tanh().forward((self.weightsinTocellMemGate.T @ self.input[i, :]) + (self.weightsHidTocellMemGate.T @ self.Hid[i, :]) + self.cellMemGatebiases)

      self.cellMem[i + 1, :] = (self.ForgetGate[i, :] * self.cellMem[i, :]) + (self.inGate[i, :] * self.cellMemGate[i, :])
      self.Hid[i + 1, :] = self.outGate[i, :] * tanh().forward(self.cellMem[i + 1, :])
      self.out[i, :] = self.outactivation.forward(self.weightsHidToout.T @ self.Hid[i + 1, :] + self.outbias)

    return self.out

  def backward(self, outError):
    inError = np.zeros(self.input.shape)
    HidError = np.zeros(self.hidshape)
    cellMemError = np.zeros(self.hidshape)

    weightsinToForgetGateΔ = np.zeros(self.weightsinToForgetGate.shape)
    weightsinToinGateΔ = np.zeros(self.weightsinToinGate.shape)
    weightsinTooutGateΔ = np.zeros(self.weightsinTooutGate.shape)
    weightsinTocellMemGateΔ = np.zeros(self.weightsinTocellMemGate.shape)

    weightsHidToForgetGateΔ = np.zeros(self.weightsHidToForgetGate.shape)
    weightsHidToinputGateΔ = np.zeros(self.weightsHidToinputGate.shape)
    weightsHidTooutputGateΔ = np.zeros(self.weightsHidTooutputGate.shape)
    weightsHidTocellMemGateΔ = np.zeros(self.weightsHidTocellMemGate.shape)

    ForgetGatebiasesΔ = np.zeros(self.ForgetGatebiases.shape)
    inGatebiasesΔ = np.zeros(self.inGatebiases.shape)
    outGatebiasesΔ = np.zeros(self.outGatebiases.shape)
    cellMemGatebiasesΔ = np.zeros(self.cellMemGatebiases.shape)

    weightsHidTooutΔ = np.zeros(self.weightsHidToout.shape)
    outbiasΔ = np.zeros(self.outbias.shape)

    for i in reversed(range(self.cellSize)):
      outGradient = self.outactivation.backward(self.out[i, :]) * outError[i, :]
      
      HidError += self.weightsHidToout @ outError[i, :]

      cellMemError += HidError * self.outGate[i] * tanh().backward(tanh().forward(self.cellMem[i + 1, :]))
      outGateError = HidError * tanh().forward(self.cellMem[i + 1, :])

      ForgetGateError = cellMemError * self.cellMem[i, :]
      inGateError = cellMemError * self.cellMemGate[i, :]
      cellMemGateError = cellMemError * self.inGate[i, :]

      cellMemError *= self.ForgetGate[i, :]

      ForgetGateGradient = sigmoid().backward(self.ForgetGate[i, :]) * ForgetGateError
      inGateGradient = sigmoid().backward(self.inGate[i, :]) * inGateError
      outGateGradient = sigmoid().backward(self.outGate[i, :]) * outGateError
      cellMemGateGradient = tanh().backward(self.cellMemGate[i, :]) * cellMemGateError

      HidError = self.weightsHidToForgetGate @ ForgetGateError + self.weightsHidToinputGate @ inGateError + self.weightsHidTooutputGate @ outGateError + self.weightsHidTocellMemGate @ cellMemGateError
      inError[i, :] = self.weightsinToForgetGate @ ForgetGateError + self.weightsinToinGate @ inGateError + self.weightsinTooutGate @ outGateError + self.weightsinTocellMemGate @ cellMemGateError

      weightsinToForgetGateΔ += np.outer(self.input[i, :].T, ForgetGateGradient)
      weightsinToinGateΔ += np.outer(self.input[i, :].T, inGateGradient)
      weightsinTooutGateΔ += np.outer(self.input[i, :].T, outGateGradient)
      weightsinTocellMemGateΔ += np.outer(self.input[i, :].T, cellMemGateGradient)

      weightsHidToForgetGateΔ += np.outer(self.Hid[i, :].T, ForgetGateGradient)
      weightsHidToinputGateΔ += np.outer(self.Hid[i, :].T, inGateGradient)
      weightsHidTooutputGateΔ += np.outer(self.Hid[i, :].T, outGateGradient)
      weightsHidTocellMemGateΔ += np.outer(self.Hid[i, :].T, cellMemGateGradient)

      ForgetGatebiasesΔ += ForgetGateGradient
      inGatebiasesΔ += inGateGradient
      outGatebiasesΔ += outGateGradient
      cellMemGatebiasesΔ += cellMemGateGradient

      weightsHidTooutΔ += np.outer(self.Hid[i].T, outGradient)
      outbiasΔ += outGradient

    self.weightsinToForgetGate += np.clip(weightsinToForgetGateΔ, -1, 1) * self.learningrate
    self.weightsinToinGate += np.clip(weightsinToinGateΔ, -1, 1) * self.learningrate
    self.weightsinTooutGate += np.clip(weightsinTooutGateΔ, -1, 1) * self.learningrate
    self.weightsinTocellMemGate += np.clip(weightsinTocellMemGateΔ, -1, 1) * self.learningrate

    self.weightsHidToForgetGate += np.clip(weightsHidToForgetGateΔ, -1, 1) * self.learningrate
    self.weightsHidToinputGate += np.clip(weightsHidToinputGateΔ, -1, 1) * self.learningrate
    self.weightsHidTooutputGate += np.clip(weightsHidTooutputGateΔ, -1, 1) * self.learningrate
    self.weightsHidTocellMemGate += np.clip(weightsHidTocellMemGateΔ, -1, 1) * self.learningrate

    self.ForgetGatebiases += np.clip(ForgetGatebiasesΔ, -1, 1) * self.learningrate
    self.inGatebiases += np.clip(inGatebiasesΔ, -1, 1) * self.learningrate
    self.outGatebiases += np.clip(outGatebiasesΔ, -1, 1) * self.learningrate
    self.cellMemGatebiases += np.clip(cellMemGatebiasesΔ, -1, 1) * self.learningrate

    self.weightsHidToout += np.clip(weightsHidTooutΔ, -1, 1) * self.learningrate
    self.outbias += np.clip(outbiasΔ, -1, 1) * self.learningrate

    return inError