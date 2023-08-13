from .tensor import tensor
from .static_ops import sigmoid, tanh, stablesoftmax

class lstm:
  def __init__(self, outshape, outactivation, hidshape=64, learningrate=0.05, inshape=None):
    self.outactivation, self.learningrate = outactivation, learningrate
    self.inshape, self.hidshape, self.outshape = inshape, hidshape, outshape
    
    if inshape is not None:
      self.weightsinToForgetGate = tensor.uniform(-0.005, 0.005, (inshape, hidshape))
      self.weightsinToinGate = tensor.uniform(-0.005, 0.005, (inshape, hidshape))
      self.weightsinTooutGate = tensor.uniform(-0.005, 0.005, (inshape, hidshape))
      self.weightsinTocellMemGate = tensor.uniform(-0.005, 0.005, (inshape, hidshape))

    self.weightshidToForgetGate = tensor.uniform(-0.005, 0.005, (hidshape, hidshape))
    self.weightshidToinputGate = tensor.uniform(-0.005, 0.005, (hidshape, hidshape))
    self.weightshidTooutputGate = tensor.uniform(-0.005, 0.005, (hidshape, hidshape))
    self.weightshidTocellMemGate = tensor.uniform(-0.005, 0.005, (hidshape, hidshape))

    self.ForgetGatebiases = tensor.zeros(hidshape)
    self.inGatebiases = tensor.zeros(hidshape)
    self.outGatebiases = tensor.zeros(hidshape)
    self.cellMemGatebiases = tensor.zeros(hidshape)

    self.weightshidToout = tensor.uniform(-0.005, 0.005, (hidshape, outshape))
    self.outbias = tensor.zeros(outshape)

  def __copy__(self):
    return type(self)(self.outshape, self.outactivation, self.hidshape, self.learningrate, self.inshape)

  def __repr__(self):
    return 'lstm(inshape=' + str(self.inshape) + ', hidshape=' + str(self.hidshape) + ', outshape=' + str(self.outshape) + ', outactivation=' + str(self.outactivation) + ', learningrate=' + str(self.learningrate) + ')'

  def modelinit(self, inshape):
    self.inshape = inshape
    self.weightsinToForgetGate = tensor.uniform(-0.005, 0.005, (inshape, self.hidshape))
    self.weightsinToinGate = tensor.uniform(-0.005, 0.005, (inshape, self.hidshape))
    self.weightsinTooutGate = tensor.uniform(-0.005, 0.005, (inshape, self.hidshape))
    self.weightsinTocellMemGate = tensor.uniform(-0.005, 0.005, (inshape, self.hidshape))
    return self.outshape

  def forward(self, input):
    self.input = input
    self.cellSize = len(input)

    self.hid = tensor.zeros((self.cellSize + 1, self.hidshape))
    self.cellMem = tensor.zeros((self.cellSize + 1, self.hidshape))
    self.out = tensor.zeros((self.cellSize, self.outshape))
    self.outderivative = tensor.zeros((self.cellSize, self.outshape))

    self.ForgetGate = tensor.zeros((self.cellSize, self.hidshape))
    self.inGate = tensor.zeros((self.cellSize, self.hidshape))
    self.outGate = tensor.zeros((self.cellSize, self.hidshape))
    self.cellMemGate = tensor.zeros((self.cellSize, self.hidshape))

    self.forgetgatederivative = tensor.zeros((self.cellSize, self.hidshape))
    self.ingatederivative = tensor.zeros((self.cellSize, self.hidshape))
    self.outgatederivative = tensor.zeros((self.cellSize, self.hidshape))
    self.cellmemgatederivative = tensor.zeros((self.cellSize, self.hidshape))

    for i in range(self.cellSize):
      self.ForgetGate[i, :] = sigmoid().forward((tensor.transpose(self.weightsinToForgetGate) @ self.input[i, :]) + (tensor.transpose(self.weightshidToForgetGate) @ self.hid[i, :]) + self.ForgetGatebiases)
      self.forgetgatederivative[i, :] = sigmoid().backward((tensor.transpose(self.weightsinToForgetGate) @ self.input[i, :]) + (tensor.transpose(self.weightshidToForgetGate) @ self.hid[i, :]) + self.ForgetGatebiases)
      
      self.inGate[i, :] = sigmoid().forward((tensor.transpose(self.weightsinToinGate) @ self.input[i, :]) + (tensor.transpose(self.weightshidToinputGate) @ self.hid[i, :]) + self.inGatebiases)
      self.ingatederivative[i, :] = sigmoid().backward((tensor.transpose(self.weightsinToinGate) @ self.input[i, :]) + (tensor.transpose(self.weightshidToinputGate) @ self.hid[i, :]) + self.inGatebiases)
      
      self.outGate[i, :] = sigmoid().forward((tensor.transpose(self.weightsinTooutGate) @ self.input[i, :]) + (tensor.transpose(self.weightshidTooutputGate) @ self.hid[i, :]) + self.outGatebiases)
      self.outgatederivative[i, :] = sigmoid().backward((tensor.transpose(self.weightsinTooutGate) @ self.input[i, :]) + (tensor.transpose(self.weightshidTooutputGate) @ self.hid[i, :]) + self.outGatebiases)
      
      self.cellMemGate[i, :] = tanh().forward((tensor.transpose(self.weightsinTocellMemGate) @ self.input[i, :]) + (tensor.transpose(self.weightshidTocellMemGate) @ self.hid[i, :]) + self.cellMemGatebiases)
      self.cellmemgatederivative[i, :] = tanh().backward((tensor.transpose(self.weightsinTocellMemGate) @ self.input[i, :]) + (tensor.transpose(self.weightshidTocellMemGate) @ self.hid[i, :]) + self.cellMemGatebiases)

      self.cellMem[i + 1, :] = (self.ForgetGate[i, :] * self.cellMem[i, :]) + (self.inGate[i, :] * self.cellMemGate[i, :])
      self.hid[i + 1, :] = self.outGate[i, :] * tanh().forward(self.cellMem[i + 1, :])
      self.out[i, :] = self.outactivation.forward(tensor.transpose(self.weightshidToout) @ self.hid[i + 1, :] + self.outbias)
      self.outderivative[i, :] = self.outactivation.backward(tensor.transpose(self.weightshidToout) @ self.hid[i + 1, :] + self.outbias)

    return self.out

  def backward(self, outError):
    inError = tensor.zeros(self.input.shape)
    hidError = tensor.zeros(self.hidshape)
    cellMemError = tensor.zeros(self.hidshape)

    weightsinToForgetGateΔ = tensor.zeros(self.weightsinToForgetGate.shape)
    weightsinToinGateΔ = tensor.zeros(self.weightsinToinGate.shape)
    weightsinTooutGateΔ = tensor.zeros(self.weightsinTooutGate.shape)
    weightsinTocellMemGateΔ = tensor.zeros(self.weightsinTocellMemGate.shape)

    weightshidToForgetGateΔ = tensor.zeros(self.weightshidToForgetGate.shape)
    weightshidToinputGateΔ = tensor.zeros(self.weightshidToinputGate.shape)
    weightshidTooutputGateΔ = tensor.zeros(self.weightshidTooutputGate.shape)
    weightshidTocellMemGateΔ = tensor.zeros(self.weightshidTocellMemGate.shape)

    ForgetGatebiasesΔ = tensor.zeros(self.ForgetGatebiases.shape)
    inGatebiasesΔ = tensor.zeros(self.inGatebiases.shape)
    outGatebiasesΔ = tensor.zeros(self.outGatebiases.shape)
    cellMemGatebiasesΔ = tensor.zeros(self.cellMemGatebiases.shape)

    weightshidTooutΔ = tensor.zeros(self.weightshidToout.shape)
    outbiasΔ = tensor.zeros(self.outbias.shape)

    for i in reversed(range(self.cellSize)):
      outGradient = self.outderivative[i, :] * outError[i, :]
      
      hidError += self.weightshidToout @ outError[i, :]

      cellMemError += hidError * self.outGate[i] * tanh().backward(self.cellMem[i + 1, :])
      outGateError = hidError * tanh().forward(self.cellMem[i + 1, :])

      ForgetGateError = cellMemError * self.cellMem[i, :]
      inGateError = cellMemError * self.cellMemGate[i, :]
      cellMemGateError = cellMemError * self.inGate[i, :]

      cellMemError *= self.ForgetGate[i, :]

      ForgetGateGradient = self.forgetgatederivative[i, :] * ForgetGateError
      inGateGradient = self.ingatederivative[i, :] * inGateError
      outGateGradient = self.outgatederivative[i, :] * outGateError
      cellMemGateGradient = self.cellmemgatederivative[i, :] * cellMemGateError

      hidError = self.weightshidToForgetGate @ ForgetGateError + self.weightshidToinputGate @ inGateError + self.weightshidTooutputGate @ outGateError + self.weightshidTocellMemGate @ cellMemGateError
      inError[i, :] = self.weightsinToForgetGate @ ForgetGateError + self.weightsinToinGate @ inGateError + self.weightsinTooutGate @ outGateError + self.weightsinTocellMemGate @ cellMemGateError

      weightsinToForgetGateΔ += tensor.outer(self.input[i, :].T, ForgetGateGradient)
      weightsinToinGateΔ += tensor.outer(self.input[i, :].T, inGateGradient)
      weightsinTooutGateΔ += tensor.outer(self.input[i, :].T, outGateGradient)
      weightsinTocellMemGateΔ += tensor.outer(self.input[i, :].T, cellMemGateGradient)

      weightshidToForgetGateΔ += tensor.outer(self.hid[i, :].T, ForgetGateGradient)
      weightshidToinputGateΔ += tensor.outer(self.hid[i, :].T, inGateGradient)
      weightshidTooutputGateΔ += tensor.outer(self.hid[i, :].T, outGateGradient)
      weightshidTocellMemGateΔ += tensor.outer(self.hid[i, :].T, cellMemGateGradient)

      ForgetGatebiasesΔ += ForgetGateGradient
      inGatebiasesΔ += inGateGradient
      outGatebiasesΔ += outGateGradient
      cellMemGatebiasesΔ += cellMemGateGradient

      weightshidTooutΔ += tensor.outer(self.hid[i].T, outGradient)
      outbiasΔ += outGradient

    self.weightsinToForgetGate += tensor.clip(weightsinToForgetGateΔ, -1, 1) * self.learningrate
    self.weightsinToinGate += tensor.clip(weightsinToinGateΔ, -1, 1) * self.learningrate
    self.weightsinTooutGate += tensor.clip(weightsinTooutGateΔ, -1, 1) * self.learningrate
    self.weightsinTocellMemGate += tensor.clip(weightsinTocellMemGateΔ, -1, 1) * self.learningrate

    self.weightshidToForgetGate += tensor.clip(weightshidToForgetGateΔ, -1, 1) * self.learningrate
    self.weightshidToinputGate += tensor.clip(weightshidToinputGateΔ, -1, 1) * self.learningrate
    self.weightshidTooutputGate += tensor.clip(weightshidTooutputGateΔ, -1, 1) * self.learningrate
    self.weightshidTocellMemGate += tensor.clip(weightshidTocellMemGateΔ, -1, 1) * self.learningrate

    self.ForgetGatebiases += tensor.clip(ForgetGatebiasesΔ, -1, 1) * self.learningrate
    self.inGatebiases += tensor.clip(inGatebiasesΔ, -1, 1) * self.learningrate
    self.outGatebiases += tensor.clip(outGatebiasesΔ, -1, 1) * self.learningrate
    self.cellMemGatebiases += tensor.clip(cellMemGatebiasesΔ, -1, 1) * self.learningrate

    self.weightshidToout += tensor.clip(weightshidTooutΔ, -1, 1) * self.learningrate
    self.outbias += tensor.clip(outbiasΔ, -1, 1) * self.learningrate

    return inError