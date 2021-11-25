import numpy as np
from .activation import *

class lstm:
  def __init__(self, inshape, outshape, outactivation, HidSize=64, learningrate=0.05):
    self.inshape, self.outshape, self.outactivation, self.HidSize, self.learningrate = inshape, outshape, outactivation, HidSize, learningrate

    self.weightsinToForgetGate = np.random.uniform(-0.005, 0.005, (inshape, HidSize))
    self.weightsinToinGate = np.random.uniform(-0.005, 0.005, (inshape, HidSize))
    self.weightsinTooutGate = np.random.uniform(-0.005, 0.005, (inshape, HidSize))
    self.weightsinTocellMemGate = np.random.uniform(-0.005, 0.005, (inshape, HidSize))

    self.weightsHidToForgetGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.weightsHidToinputGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.weightsHidTooutputGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.weightsHidTocellMemGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))

    self.ForgetGatebiases = np.zeros(HidSize)
    self.inGatebiases = np.zeros(HidSize)
    self.outGatebiases = np.zeros(HidSize)
    self.cellMemGatebiases = np.zeros(HidSize)

    self.weightsHidToout = np.random.uniform(-0.005, 0.005, (HidSize, outshape))
    self.outbias = np.zeros(outshape)

  def __copy__(self):
    return type(self)(self.inshape, self.outshape, self.outactivation, self.HidSize, self.learningrate)

  def forward(self, input):
    self.input = input
    self.cellSize = len(input)

    self.Hid = np.zeros((self.cellSize + 1, self.HidSize))
    self.cellMem = np.zeros((self.cellSize + 1, self.HidSize))
    self.out = np.zeros((self.cellSize, self.outshape))

    self.ForgetGate = np.zeros((self.cellSize, self.HidSize))
    self.inGate = np.zeros((self.cellSize, self.HidSize))
    self.outGate = np.zeros((self.cellSize, self.HidSize))
    self.cellMemGate = np.zeros((self.cellSize, self.HidSize))

    for i in range(self.cellSize):
      self.ForgetGate[i, :] = applyactivation((self.weightsinToForgetGate.T @ self.input[i, :]) + (self.weightsHidToForgetGate.T @ self.Hid[i, :]) + self.ForgetGatebiases, sigmoid())
      self.inGate[i, :] = applyactivation((self.weightsinToinGate.T @ self.input[i, :]) + (self.weightsHidToinputGate.T @ self.Hid[i, :]) + self.inGatebiases, sigmoid())
      self.outGate[i, :] = applyactivation((self.weightsinTooutGate.T @ self.input[i, :]) + (self.weightsHidTooutputGate.T @ self.Hid[i, :]) + self.outGatebiases, sigmoid())
      self.cellMemGate[i, :] = applyactivation((self.weightsinTocellMemGate.T @ self.input[i, :]) + (self.weightsHidTocellMemGate.T @ self.Hid[i, :]) + self.cellMemGatebiases, tanh())

      self.cellMem[i + 1, :] = (self.ForgetGate[i, :] * self.cellMem[i, :]) + (self.inGate[i, :] * self.cellMemGate[i, :])
      self.Hid[i + 1, :] = self.outGate[i, :] * applyactivation(self.cellMem[i + 1, :], tanh())
      self.out[i, :] = applyactivation(self.weightsHidToout.T @ self.Hid[i + 1, :] + self.outbias, self.outactivation)

    return self.out

  def backward(self, outError):
    inError = np.zeros(self.input.shape)
    HidError = np.zeros(self.HidSize)
    cellMemError = np.zeros(self.HidSize)

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
      outGradient = activationDerivative(self.out[i, :], self.outactivation) * outError[i, :]
      
      HidError += self.weightsHidToout @ outError[i, :]

      cellMemError += HidError * self.outGate[i] * activationDerivative(applyactivation(self.cellMem[i + 1, :], tanh()), tanh())
      outGateError = HidError * applyactivation(self.cellMem[i + 1, :], tanh())

      ForgetGateError = cellMemError * self.cellMem[i, :]
      inGateError = cellMemError * self.cellMemGate[i, :]
      cellMemGateError = cellMemError * self.inGate[i, :]

      cellMemError *= self.ForgetGate[i, :]

      ForgetGateGradient = activationDerivative(self.ForgetGate[i, :], sigmoid()) * ForgetGateError
      inGateGradient = activationDerivative(self.inGate[i, :], sigmoid()) * inGateError
      outGateGradient = activationDerivative(self.outGate[i, :], sigmoid()) * outGateError
      cellMemGateGradient = activationDerivative(self.cellMemGate[i, :], tanh()) * cellMemGateError

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