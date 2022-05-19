import numpy as np

class pool: 
  def __init__(self, stride, filtershape, opperation, inshape=None):
    self.stride, self.filtershape, self.opperation, self.inshape = stride, filtershape, opperation, inshape
    if inshape is not None and len(inshape) == 2:
      outshape = tuple([int(np.floor(inshape[0] / stride[0])), int(np.floor(inshape[1] / stride[1]))])
      while outshape[0] * stride[0] + filtershape[0] - stride[0] < inshape[0]:
        outshape[0] -= 1
      while outshape[1] * stride[1] + filtershape[1] - stride[1] < inshape[1]:
        outshape[1] -= 1
      self.outshape, self.out = outshape, np.zeros(outshape)
    elif inshape is not None and len(inshape) == 3:
      outshape = tuple([inshape[0], int(np.floor(inshape[1] / stride[0])), int(np.floor(inshape[2] / stride[1]))])
      while outshape[1] * stride[0] + filtershape[0] - stride[0] < inshape[1]:
        outshape[1] -= 1
      while outshape[2] * stride[1] + filtershape[1] - stride[1] < inshape[2]:
        outshape[2] -= 1
      self.outshape, self.out = outshape, np.zeros(outshape)

  def __copy__(self):
    return type(self)(self.stride, self.filtershape, self.opperation, self.inshape)

  def modelinit(self, inshape):
    self.inshape = inshape
    if len(inshape) == 2:
      outshape = tuple([int(np.floor(inshape[0] / self.stride[0])), int(np.floor(inshape[1] / self.stride[1]))])
      while outshape[0] * self.stride[0] + self.filtershape[0] - self.stride[0] < inshape[0]:
        outshape[0] -= 1
      while outshape[1] * self.stride[1] + self.filtershape[1] - self.stride[1] < inshape[1]:
        outshape[1] -= 1
    elif len(inshape) == 3:
      outshape = tuple([inshape[0], int(np.floor(inshape[1] / self.stride[0])), int(np.floor(inshape[2] / self.stride[1]))])
      while outshape[1] * self.stride[0] + self.filtershape[0] - self.stride[0] < inshape[1]:
        outshape[1] -= 1
      while outshape[2] * self.stride[1] + self.filtershape[1] - self.stride[1] < inshape[2]:
        outshape[2] -= 1
    self.outshape, self.out = outshape, np.zeros(outshape)
    return outshape

  def forward(self, input):
    self.input = input

    if self.opperation == 'Max':
      for i in range(self.outshape[1]):
        for j in range(self.outshape[2]):
          self.out[:, i, j] = np.amax(input[:, i * self.stride[0] : i * self.stride[0] + self.filtershape[0], j * self.stride[1] : j * self.stride[1] + self.filtershape[1]], axis=(1, 2))
    elif self.opperation == 'Min':
      for i in range(self.outshape[1]):
        for j in range(self.outshape[2]):
          self.out[:, i, j] = np.amin(input[:, i * self.stride[0] : i * self.stride[0] + self.filtershape[0], j * self.stride[1] : j * self.stride[1] + self.filtershape[1]], axis=(1, 2))
    elif self.opperation == 'Mean':
      for i in range(self.outshape[1]):
        for j in range(self.outshape[2]):
          self.out[:, i, j] = np.mean(input[:, i * self.stride[0] : i * self.stride[0] + self.filtershape[0], j * self.stride[1] : j * self.stride[1] + self.filtershape[1]], axis=(1, 2))
    elif self.opperation == 'Sum':
      for i in range(self.outshape[1]):
        for j in range(self.outshape[2]):
          self.out[:, i, j] = np.sum(input[:, i * self.stride[0] : i * self.stride[0] + self.filtershape[0], j * self.stride[1] : j * self.stride[1] + self.filtershape[1]], axis=(1, 2))
    return self.out

  def backward(self, outError):
    if self.opperation == 'Max' or self.opperation == 'Min':
      return np.repeat(np.repeat(outError, 2, axis=1), 2, axis=2) * np.equal(np.repeat(np.repeat(self.out, 2, axis=1), 2, axis=2), self.input).astype(int)
    elif self.opperation == 'Mean' or self.opperation == 'Sum':
      return np.repeat(np.repeat(outError, 2, axis=1), 2, axis=2) * np.repeat(np.repeat(self.out, 2, axis=1), 2, axis=2)