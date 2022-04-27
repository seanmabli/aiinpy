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

class gru:
  def __init__(self, outshape, outactivation, hidshape=64, learningrate=0.05, inshape=None):
    self.outactivation, self.learningrate = outactivation, learningrate
    self.inshape, self.hidshape, self.outshape = inshape, hidshape, outshape

    if inshape is not None:
      self.weightsinToResetGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
      self.weightsinToUpdateGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
      self.weightsinTohidGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))

    self.weightshidToResetGate = np.random.uniform(-0.005, 0.005, (hidshape, hidshape))
    self.weightshidToUpdateGate = np.random.uniform(-0.005, 0.005, (hidshape, hidshape))
    self.weightshidTohidGate = np.random.uniform(-0.005, 0.005, (hidshape, hidshape))

    self.hidGatebias = np.zeros(hidshape)
    self.ResetGatebias = np.zeros(hidshape)
    self.UpdateGatebias = np.zeros(hidshape)

    self.weightshidToout = np.random.uniform(-0.005, 0.005, (hidshape, outshape))
    self.outbias = np.zeros(outshape)

  def __copy__(self):
    return type(self)(self.outshape, self.outactivation, self.hidshape, self.learningrate, self.inshape)

  def modelinit(self, inshape):
    self.inshape = inshape
    self.weightsinToResetGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
    self.weightsinToUpdateGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
    self.weightsinTohidGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
    return self.outshape

  def forward(self, input):
    self.input = input
    self.cellSize = len(input)

    self.hid = np.zeros((self.cellSize + 1, self.hidshape))
    self.out = np.zeros((self.cellSize, self.outshape))
  
    self.ResetGate = np.zeros((self.cellSize, self.hidshape))
    self.UpdateGate = np.zeros((self.cellSize, self.hidshape))
    self.hidGate = np.zeros((self.cellSize, self.hidshape))

    for i in range(self.cellSize):
      self.ResetGate[i, :] = sigmoid().forward(self.weightsinToResetGate.T @ self.input[i, :] + self.weightshidToResetGate.T @ self.hid[i, :] + self.ResetGatebias)
      self.UpdateGate[i, :] = sigmoid().forward(self.weightsinToUpdateGate.T @ self.input[i, :] + self.weightshidToUpdateGate.T @ self.hid[i, :] + self.UpdateGatebias)
      self.hidGate[i, :] = tanh().forward(self.weightsinTohidGate.T @ self.input[i, :] + self.weightshidTohidGate.T @ (self.hid[i, :] * self.ResetGate[i, :]) + self.hidGatebias)
  
      self.hid[i + 1, :] = (1 - self.UpdateGate[i, :]) * self.hid[i, :] + self.UpdateGate[i, :] * self.hidGate[i, :]
      self.out[i, :] = self.outactivation.forward(self.weightshidToout.T @ self.hid[i + 1, :] + self.outbias)

    return self.out

  def backward(self, outError):
    inError = np.zeros(self.input.shape)
    hidError = np.zeros(self.hidshape)

    weightsinToResetGateΔ = np.zeros(self.weightsinToResetGate.shape)
    weightsinToUpdateGateΔ = np.zeros(self.weightsinToUpdateGate.shape)
    weightsinTohidGateΔ = np.zeros(self.weightsinTohidGate.shape)

    weightshidToResetGateΔ = np.zeros(self.weightshidToResetGate.shape)
    weightshidToUpdateGateΔ = np.zeros(self.weightshidToUpdateGate.shape)
    weightshidTohidGateΔ = np.zeros(self.weightshidTohidGate.shape)

    hidGatebiasΔ = np.zeros(self.hidGatebias.shape)
    ResetGatebiasΔ = np.zeros(self.ResetGatebias.shape)
    UpdateGatebiasΔ = np.zeros(self.UpdateGatebias.shape)

    weightshidTooutΔ = np.zeros(self.weightshidToout.shape)
    outbiasΔ = np.zeros(self.outbias.shape)

    for i in reversed(range(self.cellSize)):
      outGradient = self.outactivation.backward(self.out[i, :]) * outError[i, :]

      hidError += self.weightshidToout @ outError[i, :]

      hidGateError = hidError * self.UpdateGate[i, :]
      UpdateGateError = hidError * (-1 * self.hid[i, :]) + hidError * self.hidGate[i, :]
      ResetGateError = (self.weightshidTohidGate.T @ hidGateError) * self.hid[i, :]

      hidError += (self.weightshidTohidGate.T @ hidGateError) * self.ResetGate[i, :] + self.weightshidToResetGate.T @ ResetGateError + self.weightshidToUpdateGate.T @ UpdateGateError
      inError[i, :] = self.weightsinToResetGate @ ResetGateError + self.weightsinToUpdateGate @ UpdateGateError + self.weightsinTohidGate @ hidGateError

      ResetGateGradient = sigmoid().backward(self.ResetGate[i, :]) * ResetGateError
      UpdateGateGradient = sigmoid().backward(self.UpdateGate[i, :]) * UpdateGateError
      hidGateGradient = tanh().backward(self.hidGate[i, :]) * hidGateError

      weightsinToResetGateΔ += np.outer(self.input[i, :].T, ResetGateGradient)
      weightsinToUpdateGateΔ += np.outer(self.input[i, :].T, UpdateGateGradient)
      weightsinTohidGateΔ += np.outer(self.input[i, :].T, hidGateGradient)
  
      weightshidToResetGateΔ += np.outer(self.hid[i, :].T, ResetGateGradient)
      weightshidToUpdateGateΔ += np.outer(self.hid[i, :].T, UpdateGateGradient)
      weightshidTohidGateΔ += np.outer(self.hid[i, :].T, hidGateGradient)
  
      hidGatebiasΔ += ResetGateGradient
      ResetGatebiasΔ += UpdateGateGradient
      UpdateGatebiasΔ += hidGateGradient

      weightshidTooutΔ += np.outer(self.hid[i].T, outGradient)
      outbiasΔ += outGradient

    self.weightsinToResetGate += np.clip(weightsinToResetGateΔ, -1, 1) * self.learningrate
    self.weightsinToUpdateGate += np.clip(weightsinToUpdateGateΔ, -1, 1) * self.learningrate
    self.weightsinTohidGate += np.clip(weightsinTohidGateΔ, -1, 1) * self.learningrate

    self.weightshidToResetGate += np.clip(weightshidToResetGateΔ, -1, 1) * self.learningrate
    self.weightshidToUpdateGate += np.clip(weightshidToUpdateGateΔ, -1, 1) * self.learningrate
    self.weightshidTohidGate += np.clip(weightshidTohidGateΔ, -1, 1) * self.learningrate

    self.hidGatebias += np.clip(hidGatebiasΔ, -1, 1) * self.learningrate
    self.ResetGatebias += np.clip(ResetGatebiasΔ, -1, 1) * self.learningrate
    self.UpdateGatebias += np.clip(UpdateGatebiasΔ, -1, 1) * self.learningrate

    self.weightshidToout += np.clip(weightshidTooutΔ, -1, 1) * self.learningrate
    self.outbias += np.clip(outbiasΔ, -1, 1) * self.learningrate

    return inError