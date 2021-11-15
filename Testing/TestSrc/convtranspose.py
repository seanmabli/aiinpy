import numpy as np
from .activation import *

class convtranspose:
  def __init__(self, InShape, FilterShape, LearningRate, Activation, Padding=False, Stride=(1, 1)):
    self.InShape, self.FilterShape, self.LearningRate, self.Activation, self.Padding, self.Stride = InShape, FilterShape, LearningRate, Activation, Padding, Stride
    if len(InShape) == 2:
      InShape = tuple([self.FilterShape[0]]) + InShape
    self.OutShape = np.array([FilterShape[0], (((InShape[1] + 1) * FilterShape[1]) / Stride[0]) - (2 * Padding), (((InShape[2] + 1) * FilterShape[2]) / Stride[1]) - (2 * Padding)], dtype=np.int)
    self.Out = np.zeros(self.OutShape)

    self.Filter = np.random.uniform(-0.25, 0.25, (self.FilterShape))
    self.Bias = np.zeros(self.FilterShape[0])

  def forward(self, In):
    self.In = In
    if(In.ndim == 2):
      self.In = np.stack(([self.In] * self.FilterShape[0]))

    for i in range(0, self.OutShape[1], self.Stride[0]):
      for j in range(0, self.OutShape[2], self.Stride[1]):
        self.Out[:, i : i + FilterShape[1], j : j  + FilterShape[2]] += np.multiply(self.In[:, i, j], self.Filter)

    self.Out += self.Bias[:, np.newaxis, np.newaxis]
    self.Out = ApplyActivation(self.Out, self.Activation)

    if self.Padding == True:
      self.Out = self.Out[:, 1 : self.OutShape[1] - 1, 1 : self.OutShape[2] - 1]

    return self.Out

  def backward(self, OutError):
    FilterΔ = np.zeros(self.FilterShape)

    OutGradient = ActivationDerivative(self.Out, self.Activation) * OutError
    OutGradient = np.pad(OutGradient, 1, mode='constant')[1 : self.FilterShape[0] + 1, :, :]

    for i in range(0, self.OutShape[1], self.Stride[0]):
      for j in range(0, self.OutShape[2], self.Stride[1]):
        FilterΔ += self.In[:, i, j] * OutGradient[:, i : i + FilterShape[1], j : j  + FilterShape[2]]

    self.Bias += np.sum(OutGradient, axis=(1, 2)) * self.LearningRate
    self.Filter += FilterΔ * self.LearningRate

    # In Error
    RotFilter = np.rot90(np.rot90(self.Filter))
    PaddedError = np.pad(OutError, self.FilterShape[1] - 1, mode='constant')[self.FilterShape[1] - 1 : self.FilterShape[0] + self.FilterShape[1] - 1, :, :]
    
    self.InError = np.zeros(self.InShape)
    if np.ndim(self.InError) == 3:
      for i in range(int(self.InShape[1] / self.Stride[0])):
        for j in range(int(self.InShape[2] / self.Stride[1])):
         self.InError[:, i * self.Stride[0], j * self.Stride[1]] = np.sum(np.multiply(RotFilter, PaddedError[:, i:i + 3, j:j + 3]), axis=(1, 2))
       
    if np.ndim(self.InError) == 2:
      for i in range(int(self.InShape[0] / self.Stride[0])):
        for j in range(int(self.InShape[1] / self.Stride[1])):
         self.InError[i * self.Stride[0], j * self.Stride[1]] = np.sum(np.multiply(RotFilter, PaddedError[:, i:i + 3, j:j + 3]))

    return self.InError