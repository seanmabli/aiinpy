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
      self.weightsinToHidGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))

    self.weightsHidToResetGate = np.random.uniform(-0.005, 0.005, (hidshape, hidshape))
    self.weightsHidToUpdateGate = np.random.uniform(-0.005, 0.005, (hidshape, hidshape))
    self.weightsHidToHidGate = np.random.uniform(-0.005, 0.005, (hidshape, hidshape))

    self.HidGatebias = np.zeros(hidshape)
    self.ResetGatebias = np.zeros(hidshape)
    self.UpdateGatebias = np.zeros(hidshape)

    self.weightsHidToout = np.random.uniform(-0.005, 0.005, (hidshape, outshape))
    self.outbias = np.zeros(outshape)

  def __copy__(self):
    return type(self)(self.outshape, self.outactivation, self.hidshape, self.learningrate, self.inshape)

  def modelinit(self, inshape):
    self.inshape = inshape
    self.weightsinToResetGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
    self.weightsinToUpdateGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
    self.weightsinToHidGate = np.random.uniform(-0.005, 0.005, (inshape, hidshape))
    return self.outshape

  def forward(self, input):
    self.input = input
    self.cellSize = len(input)

    self.Hid = np.zeros((self.cellSize + 1, self.hidshape))
    self.out = np.zeros((self.cellSize, self.outshape))
  
    self.ResetGate = np.zeros((self.cellSize, self.hidshape))
    self.UpdateGate = np.zeros((self.cellSize, self.hidshape))
    self.HidGate = np.zeros((self.cellSize, self.hidshape))

    for i in range(self.cellSize):
      self.ResetGate[i, :] = sigmoid().forward(self.weightsinToResetGate.T @ self.input[i, :] + self.weightsHidToResetGate.T @ self.Hid[i, :] + self.ResetGatebias)
      self.UpdateGate[i, :] = sigmoid().forward(self.weightsinToUpdateGate.T @ self.input[i, :] + self.weightsHidToUpdateGate.T @ self.Hid[i, :] + self.UpdateGatebias)
      self.HidGate[i, :] = tanh().forward(self.weightsinToHidGate.T @ self.input[i, :] + self.weightsHidToHidGate.T @ (self.Hid[i, :] * self.ResetGate[i, :]) + self.HidGatebias)
  
      self.Hid[i + 1, :] = (1 - self.UpdateGate[i, :]) * self.Hid[i, :] + self.UpdateGate[i, :] * self.HidGate[i, :]
      self.out[i, :] = self.outactivation.forward(self.weightsHidToout.T @ self.Hid[i + 1, :] + self.outbias)

    return self.out

  def backward(self, outError):
    inError = np.zeros(self.input.shape)
    HidError = np.zeros(self.hidshape)

    weightsinToResetGateΔ = np.zeros(self.weightsinToResetGate.shape)
    weightsinToUpdateGateΔ = np.zeros(self.weightsinToUpdateGate.shape)
    weightsinToHidGateΔ = np.zeros(self.weightsinToHidGate.shape)

    weightsHidToResetGateΔ = np.zeros(self.weightsHidToResetGate.shape)
    weightsHidToUpdateGateΔ = np.zeros(self.weightsHidToUpdateGate.shape)
    weightsHidToHidGateΔ = np.zeros(self.weightsHidToHidGate.shape)

    HidGatebiasΔ = np.zeros(self.HidGatebias.shape)
    ResetGatebiasΔ = np.zeros(self.ResetGatebias.shape)
    UpdateGatebiasΔ = np.zeros(self.UpdateGatebias.shape)

    weightsHidTooutΔ = np.zeros(self.weightsHidToout.shape)
    outbiasΔ = np.zeros(self.outbias.shape)

    for i in reversed(range(self.cellSize)):
      outGradient = self.outactivation.backward(self.out[i, :]) * outError[i, :]

      HidError += self.weightsHidToout @ outError[i, :]

      HidGateError = HidError * self.UpdateGate[i, :]
      UpdateGateError = HidError * (-1 * self.Hid[i, :]) + HidError * self.HidGate[i, :]
      ResetGateError = (self.weightsHidToHidGate.T @ HidGateError) * self.Hid[i, :]

      HidError += (self.weightsHidToHidGate.T @ HidGateError) * self.ResetGate[i, :] + self.weightsHidToResetGate.T @ ResetGateError + self.weightsHidToUpdateGate.T @ UpdateGateError
      inError[i, :] = self.weightsinToResetGate @ ResetGateError + self.weightsinToUpdateGate @ UpdateGateError + self.weightsinToHidGate @ HidGateError

      ResetGateGradient = sigmoid().backward(self.ResetGate[i, :]) * ResetGateError
      UpdateGateGradient = sigmoid().backward(self.UpdateGate[i, :]) * UpdateGateError
      HidGateGradient = tanh().backward(self.HidGate[i, :]) * HidGateError

      weightsinToResetGateΔ += np.outer(self.input[i, :].T, ResetGateGradient)
      weightsinToUpdateGateΔ += np.outer(self.input[i, :].T, UpdateGateGradient)
      weightsinToHidGateΔ += np.outer(self.input[i, :].T, HidGateGradient)
  
      weightsHidToResetGateΔ += np.outer(self.Hid[i, :].T, ResetGateGradient)
      weightsHidToUpdateGateΔ += np.outer(self.Hid[i, :].T, UpdateGateGradient)
      weightsHidToHidGateΔ += np.outer(self.Hid[i, :].T, HidGateGradient)
  
      HidGatebiasΔ += ResetGateGradient
      ResetGatebiasΔ += UpdateGateGradient
      UpdateGatebiasΔ += HidGateGradient

      weightsHidTooutΔ += np.outer(self.Hid[i].T, outGradient)
      outbiasΔ += outGradient

    self.weightsinToResetGate += np.clip(weightsinToResetGateΔ, -1, 1) * self.learningrate
    self.weightsinToUpdateGate += np.clip(weightsinToUpdateGateΔ, -1, 1) * self.learningrate
    self.weightsinToHidGate += np.clip(weightsinToHidGateΔ, -1, 1) * self.learningrate

    self.weightsHidToResetGate += np.clip(weightsHidToResetGateΔ, -1, 1) * self.learningrate
    self.weightsHidToUpdateGate += np.clip(weightsHidToUpdateGateΔ, -1, 1) * self.learningrate
    self.weightsHidToHidGate += np.clip(weightsHidToHidGateΔ, -1, 1) * self.learningrate

    self.HidGatebias += np.clip(HidGatebiasΔ, -1, 1) * self.learningrate
    self.ResetGatebias += np.clip(ResetGatebiasΔ, -1, 1) * self.learningrate
    self.UpdateGatebias += np.clip(UpdateGatebiasΔ, -1, 1) * self.learningrate

    self.weightsHidToout += np.clip(weightsHidTooutΔ, -1, 1) * self.learningrate
    self.outbias += np.clip(outbiasΔ, -1, 1) * self.learningrate

    return inError