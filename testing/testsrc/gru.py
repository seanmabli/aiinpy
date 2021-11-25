import numpy as np
from .activation import *

class gru:
  def __init__(self, inshape, outshape, outactivation, HidSize=64, learningrate=0.05):
    self.inshape, self.outshape, self.outactivation, self.HidSize, self.learningrate = inshape, outshape, outactivation, HidSize, learningrate

    self.weightsinToResetGate = np.random.uniform(-0.005, 0.005, (inshape, HidSize))
    self.weightsinToUpdateGate = np.random.uniform(-0.005, 0.005, (inshape, HidSize))
    self.weightsinToHidGate = np.random.uniform(-0.005, 0.005, (inshape, HidSize))

    self.weightsHidToResetGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.weightsHidToUpdateGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))
    self.weightsHidToHidGate = np.random.uniform(-0.005, 0.005, (HidSize, HidSize))

    self.HidGatebias = np.zeros(HidSize)
    self.ResetGatebias = np.zeros(HidSize)
    self.UpdateGatebias = np.zeros(HidSize)

    self.weightsHidToout = np.random.uniform(-0.005, 0.005, (HidSize, outshape))
    self.outbias = np.zeros(outshape)

  def __copy__(self):
    return type(self)(self.inshape, self.outshape, self.outactivation, self.HidSize, self.learningrate)

  def modelinit(self, inshape):
    pass

  def forward(self, input):
    self.input = input
    self.cellSize = len(input)

    self.Hid = np.zeros((self.cellSize + 1, self.HidSize))
    self.out = np.zeros((self.cellSize, self.outshape))
  
    self.ResetGate = np.zeros((self.cellSize, self.HidSize))
    self.UpdateGate = np.zeros((self.cellSize, self.HidSize))
    self.HidGate = np.zeros((self.cellSize, self.HidSize))

    for i in range(self.cellSize):
      self.ResetGate[i, :] = applyactivation(self.weightsinToResetGate.T @ self.input[i, :] + self.weightsHidToResetGate.T @ self.Hid[i, :] + self.ResetGatebias, sigmoid())
      self.UpdateGate[i, :] = applyactivation(self.weightsinToUpdateGate.T @ self.input[i, :] + self.weightsHidToUpdateGate.T @ self.Hid[i, :] + self.UpdateGatebias, sigmoid())
      self.HidGate[i, :] = applyactivation(self.weightsinToHidGate.T @ self.input[i, :] + self.weightsHidToHidGate.T @ (self.Hid[i, :] * self.ResetGate[i, :]) + self.HidGatebias, tanh())
  
      self.Hid[i + 1, :] = (1 - self.UpdateGate[i, :]) * self.Hid[i, :] + self.UpdateGate[i, :] * self.HidGate[i, :]
      self.out[i, :] = applyactivation(self.weightsHidToout.T @ self.Hid[i + 1, :] + self.outbias, self.outactivation)

    return self.out

  def backward(self, outError):
    inError = np.zeros(self.input.shape)
    HidError = np.zeros(self.HidSize)

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
      outGradient = activationDerivative(self.out[i, :], self.outactivation) * outError[i, :]

      HidError += self.weightsHidToout @ outError[i, :]

      HidGateError = HidError * self.UpdateGate[i, :]
      UpdateGateError = HidError * (-1 * self.Hid[i, :]) + HidError * self.HidGate[i, :]
      ResetGateError = (self.weightsHidToHidGate.T @ HidGateError) * self.Hid[i, :]

      HidError += (self.weightsHidToHidGate.T @ HidGateError) * self.ResetGate[i, :] + self.weightsHidToResetGate.T @ ResetGateError + self.weightsHidToUpdateGate.T @ UpdateGateError
      inError[i, :] = self.weightsinToResetGate @ ResetGateError + self.weightsinToUpdateGate @ UpdateGateError + self.weightsinToHidGate @ HidGateError

      ResetGateGradient = activationDerivative(self.ResetGate[i, :], sigmoid()) * ResetGateError
      UpdateGateGradient = activationDerivative(self.UpdateGate[i, :], sigmoid()) * UpdateGateError
      HidGateGradient = activationDerivative(self.HidGate[i, :], tanh()) * HidGateError

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