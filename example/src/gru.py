from .static_ops import sigmoid, tanh

class gru:
  def __init__(self, outshape, outactivation, hidshape=64, learningrate=0.05, inshape=None):
    self.outactivation, self.learningrate = outactivation, learningrate
    self.inshape, self.hidshape, self.outshape = inshape, hidshape, outshape

    if inshape is not None:
      self.weightsinToResetGate = tensor.uniform(-0.005, 0.005, (inshape, hidshape))
      self.weightsinToUpdateGate = tensor.uniform(-0.005, 0.005, (inshape, hidshape))
      self.weightsinTohidGate = tensor.uniform(-0.005, 0.005, (inshape, hidshape))

    self.weightshidToResetGate = tensor.uniform(-0.005, 0.005, (hidshape, hidshape))
    self.weightshidToUpdateGate = tensor.uniform(-0.005, 0.005, (hidshape, hidshape))
    self.weightshidTohidGate = tensor.uniform(-0.005, 0.005, (hidshape, hidshape))

    self.hidGatebias = tensor.zeros(hidshape)
    self.ResetGatebias = tensor.zeros(hidshape)
    self.UpdateGatebias = tensor.zeros(hidshape)

    self.weightshidToout = tensor.uniform(-0.005, 0.005, (hidshape, outshape))
    self.outbias = tensor.zeros(outshape)

  def __copy__(self):
    return type(self)(self.outshape, self.outactivation, self.hidshape, self.learningrate, self.inshape)

  def __repr__(self):
    return 'gru(inshape=' + str(self.inshape) + ', hidshape=' + str(self.hidshape) + ', outshape=' + str(self.outshape) + ', outactivation=' + str(self.outactivation) + ', learningrate=' + str(self.learningrate) + ')'

  def modelinit(self, inshape):
    self.inshape = inshape
    self.weightsinToResetGate = tensor.uniform(-0.005, 0.005, (inshape, hidshape))
    self.weightsinToUpdateGate = tensor.uniform(-0.005, 0.005, (inshape, hidshape))
    self.weightsinTohidGate = tensor.uniform(-0.005, 0.005, (inshape, hidshape))
    return self.outshape

  def forward(self, itensorut):
    self.itensorut = itensorut
    self.cellSize = len(itensorut)

    self.hid = tensor.zeros((self.cellSize + 1, self.hidshape))
    self.out = tensor.zeros((self.cellSize, self.outshape))
    self.outderivative = tensor.zeros((self.cellSize, self.outshape))
  
    self.ResetGate = tensor.zeros((self.cellSize, self.hidshape))
    self.UpdateGate = tensor.zeros((self.cellSize, self.hidshape))
    self.hidGate = tensor.zeros((self.cellSize, self.hidshape))

    self.resetgatederivative = tensor.zeros((self.cellSize, self.hidshape))
    self.updategatederivative = tensor.zeros((self.cellSize, self.hidshape))
    self.hidgatederivative = tensor.zeros((self.cellSize, self.hidshape))


    for i in range(self.cellSize):
      self.ResetGate[i, :] = sigmoid().forward(self.weightsinToResetGate.T @ self.itensorut[i, :] + self.weightshidToResetGate.T @ self.hid[i, :] + self.ResetGatebias)
      self.resetgatederivative[i, :] = sigmoid().backward(self.weightsinToResetGate.T @ self.itensorut[i, :] + self.weightshidToResetGate.T @ self.hid[i, :] + self.ResetGatebias)
      self.UpdateGate[i, :] = sigmoid().forward(self.weightsinToUpdateGate.T @ self.itensorut[i, :] + self.weightshidToUpdateGate.T @ self.hid[i, :] + self.UpdateGatebias)
      self.updategatederivative[i, :] = sigmoid().backward(self.weightsinToUpdateGate.T @ self.itensorut[i, :] + self.weightshidToUpdateGate.T @ self.hid[i, :] + self.UpdateGatebias)
      self.hidGate[i, :] = tanh().forward(self.weightsinTohidGate.T @ self.itensorut[i, :] + self.weightshidTohidGate.T @ (self.hid[i, :] * self.ResetGate[i, :]) + self.hidGatebias)
      self.hidgatederivative[i, :] = tanh().backward(self.weightsinTohidGate.T @ self.itensorut[i, :] + self.weightshidTohidGate.T @ (self.hid[i, :] * self.ResetGate[i, :]) + self.hidGatebias)

      self.hid[i + 1, :] = (1 - self.UpdateGate[i, :]) * self.hid[i, :] + self.UpdateGate[i, :] * self.hidGate[i, :]
      self.out[i, :] = self.outactivation.forward(self.weightshidToout.T @ self.hid[i + 1, :] + self.outbias)
      self.outderivative[i, :] = self.outactivation.backward(self.weightshidToout.T @ self.hid[i + 1, :] + self.outbias)
    
    return self.out

  def backward(self, outError):
    inError = tensor.zeros(self.itensorut.shape)
    hidError = tensor.zeros(self.hidshape)

    weightsinToResetGateΔ = tensor.zeros(self.weightsinToResetGate.shape)
    weightsinToUpdateGateΔ = tensor.zeros(self.weightsinToUpdateGate.shape)
    weightsinTohidGateΔ = tensor.zeros(self.weightsinTohidGate.shape)

    weightshidToResetGateΔ = tensor.zeros(self.weightshidToResetGate.shape)
    weightshidToUpdateGateΔ = tensor.zeros(self.weightshidToUpdateGate.shape)
    weightshidTohidGateΔ = tensor.zeros(self.weightshidTohidGate.shape)

    hidGatebiasΔ = tensor.zeros(self.hidGatebias.shape)
    ResetGatebiasΔ = tensor.zeros(self.ResetGatebias.shape)
    UpdateGatebiasΔ = tensor.zeros(self.UpdateGatebias.shape)

    weightshidTooutΔ = tensor.zeros(self.weightshidToout.shape)
    outbiasΔ = tensor.zeros(self.outbias.shape)

    for i in reversed(range(self.cellSize)):
      outGradient = self.outderivative[i, :] * outError[i, :]

      hidError += self.weightshidToout @ outError[i, :]

      hidGateError = hidError * self.UpdateGate[i, :]
      UpdateGateError = hidError * (-1 * self.hid[i, :]) + hidError * self.hidGate[i, :]
      ResetGateError = (self.weightshidTohidGate.T @ hidGateError) * self.hid[i, :]

      hidError += (self.weightshidTohidGate.T @ hidGateError) * self.ResetGate[i, :] + self.weightshidToResetGate.T @ ResetGateError + self.weightshidToUpdateGate.T @ UpdateGateError
      inError[i, :] = self.weightsinToResetGate @ ResetGateError + self.weightsinToUpdateGate @ UpdateGateError + self.weightsinTohidGate @ hidGateError

      ResetGateGradient = self.resetgatederivative[i, :] * ResetGateError
      UpdateGateGradient = self.updategatederivative[i, :] * UpdateGateError
      hidGateGradient = self.hidgatederivative[i, :] * hidGateError

      weightsinToResetGateΔ += tensor.outer(self.itensorut[i, :].T, ResetGateGradient)
      weightsinToUpdateGateΔ += tensor.outer(self.itensorut[i, :].T, UpdateGateGradient)
      weightsinTohidGateΔ += tensor.outer(self.itensorut[i, :].T, hidGateGradient)
  
      weightshidToResetGateΔ += tensor.outer(self.hid[i, :].T, ResetGateGradient)
      weightshidToUpdateGateΔ += tensor.outer(self.hid[i, :].T, UpdateGateGradient)
      weightshidTohidGateΔ += tensor.outer(self.hid[i, :].T, hidGateGradient)
  
      hidGatebiasΔ += ResetGateGradient
      ResetGatebiasΔ += UpdateGateGradient
      UpdateGatebiasΔ += hidGateGradient

      weightshidTooutΔ += tensor.outer(self.hid[i].T, outGradient)
      outbiasΔ += outGradient

    self.weightsinToResetGate += tensor.clip(weightsinToResetGateΔ, -1, 1) * self.learningrate
    self.weightsinToUpdateGate += tensor.clip(weightsinToUpdateGateΔ, -1, 1) * self.learningrate
    self.weightsinTohidGate += tensor.clip(weightsinTohidGateΔ, -1, 1) * self.learningrate

    self.weightshidToResetGate += tensor.clip(weightshidToResetGateΔ, -1, 1) * self.learningrate
    self.weightshidToUpdateGate += tensor.clip(weightshidToUpdateGateΔ, -1, 1) * self.learningrate
    self.weightshidTohidGate += tensor.clip(weightshidTohidGateΔ, -1, 1) * self.learningrate

    self.hidGatebias += tensor.clip(hidGatebiasΔ, -1, 1) * self.learningrate
    self.ResetGatebias += tensor.clip(ResetGatebiasΔ, -1, 1) * self.learningrate
    self.UpdateGatebias += tensor.clip(UpdateGatebiasΔ, -1, 1) * self.learningrate

    self.weightshidToout += tensor.clip(weightshidTooutΔ, -1, 1) * self.learningrate
    self.outbias += tensor.clip(outbiasΔ, -1, 1) * self.learningrate

    return inError