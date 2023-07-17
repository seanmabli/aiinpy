import numpy as np
import math
class convtranspose:
  def __init__(self, inshape, filtershape, learningrate, padding=False, stride=(1, 1)):
    filtershape = tuple([1]) + filtershape if len(filtershape) == 2 else filtershape
    self.inshape, self.filtershape, self.learningrate, self.padding, self.stride = inshape, filtershape, learningrate, padding, stride
    if len(inshape) == 2:
      self.inshape = inshape = tuple([self.filtershape[0]]) + inshape
    fakepadding = (0 if stride[0] > filtershape[1] else filtershape[1] - stride[0], 0 if stride[1] > filtershape[2] else filtershape[2] - stride[1])
    padding = (0 if padding == True or stride[0] > filtershape[1] else filtershape[1] - stride[0], 0 if padding == True or stride[1] > filtershape[2] else filtershape[2] - stride[1])

    self.fakeoutshape = (filtershape[0], inshape[1] * stride[0] + fakepadding[0], inshape[2] * stride[1] + fakepadding[1])
    self.outshape = (filtershape[0], inshape[1] * stride[0] + padding[0], inshape[2] * stride[1] + padding[1])

    self.Filter = np.random.uniform(-0.25, 0.25, (self.filtershape))
    self.bias = np.zeros(self.filtershape[0])

  def __repr__(self):
    return 'convtranspose(inshape=' + str(self.inshape) + ', outshape=' + str(self.outshape) + ', filtershape=' + str(self.filtershape) + ', learningrate=' + str(self.learningrate) + ', padding=' + str(self.padding) + ', stride=' + str(self.stride) + ')'

  def modelinit(self, inshape):
    return self.outshape

  def forward(self, input):
    self.input = input
    if(input.ndim == 2):
      self.input = np.stack(([self.input] * self.filtershape[0]))

    self.out = np.zeros(self.fakeoutshape)
    for i in range(0, self.inshape[1]):
      for j in range(0, self.inshape[2]):
        self.out[:, i * self.stride[0] : i * self.stride[0] + self.filtershape[1], j * self.stride[1] : j * self.stride[1] + self.filtershape[2]] += self.input[:, i, j][:, np.newaxis, np.newaxis] * self.Filter

    self.out += self.bias[:, np.newaxis, np.newaxis]

    if self.padding:
      paddingdifference = tuple(map(lambda i, j: i - j, self.fakeoutshape, self.outshape))
      self.out = self.out[:, math.floor(paddingdifference[1] / 2) : self.fakeoutshape[1] - math.ceil(paddingdifference[1] / 2), math.floor(paddingdifference[2] / 2) : self.fakeoutshape[2] - math.ceil(paddingdifference[2] / 2)]
    
    return self.out

  def backward(self, outError):
    FilterΔ = np.zeros(self.filtershape)

    outGradient = self.derivative * outError
    outGradient = np.pad(outGradient, 1, mode='constant')[1 : self.filtershape[0] + 1, :, :]

    for i in range(0, self.inshape[1]):
      for j in range(0, self.inshape[2]):
        FilterΔ += self.input[:, i, j][:, np.newaxis, np.newaxis] * outGradient[:, i * self.stride[0] : i * self.stride[0] + self.filtershape[1], j * self.stride[1] : j * self.stride[1] + self.filtershape[2]]

    self.bias += np.sum(outGradient, axis=(1, 2)) * self.learningrate
    self.Filter += FilterΔ * self.learningrate

    # in Error
    RotFilter = np.rot90(np.rot90(self.Filter))
    PaddedError = np.pad(outError, self.filtershape[1] - 1, mode='constant')[self.filtershape[1] - 1 : self.filtershape[0] + self.filtershape[1] - 1, :, :]
    
    self.inError = np.zeros(self.inshape)
    if np.ndim(self.inError) == 3:
      for i in range(int(self.inshape[1] / self.stride[0])):
        for j in range(int(self.inshape[2] / self.stride[1])):
         self.inError[:, i * self.stride[0], j * self.stride[1]] = np.sum(np.multiply(RotFilter, PaddedError[:, i:i + self.filtershape[1], j:j + self.filtershape[2]]), axis=(1, 2))
       
    if np.ndim(self.inError) == 2:
      for i in range(int(self.inshape[0] / self.stride[0])):
        for j in range(int(self.inshape[1] / self.stride[1])):
         self.inError[i * self.stride[0], j * self.stride[1]] = np.sum(np.multiply(RotFilter, PaddedError[:, i:i + self.filtershape[1], j:j + self.filtershape[2]]))

    return self.inError