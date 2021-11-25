import numpy as np

class pool: 
  def __init__(self, inshape, stride, filtershape, opperation):
    self.stride, self.filtershape, self.opperation = stride, filtershape, opperation
    if len(inshape) == 2:
      self.out = np.zeros((int(np.floor(inshape[0] / self.stride[0])), int(np.floor(inshape[1] / self.stride[1]))))
      while self.out.shape[0] * self.stride[0] + self.filtershape[0] - self.stride[0] < inshape[0]:
        self.out = self.out[ : - 1, :]
      while self.out.shape[1] * self.stride[1] + self.filtershape[1] - self.stride[1] < inshape[1]:
        self.out = self.out[:, : - 1]
    elif len(inshape) == 3:
      self.out = np.zeros((inshape[0], int(np.floor(inshape[1] / self.stride[0])), int(np.floor(inshape[2] / self.stride[1]))))
      while self.out.shape[1] * self.stride[0] + self.filtershape[0] - self.stride[0] < inshape[1]:
        self.out = self.out[:, : - 1, :]
      while self.out.shape[2] * self.stride[1] + self.filtershape[1] - self.stride[1] < inshape[2]:
        self.out = self.out[:, :, : - 1]

  def __copy__(self):
    return type(self)(self.stride, self.filtershape, self.opperation)

  def modelinit(self, inshape):
    pass

  def forward(self, input):
    self.input = input

    if self.opperation == 'Max':
      for i in range(self.out.shape[1]):
        for j in range(self.out.shape[2]):
          self.out[:, i, j] = np.amax(input[:, i * self.stride[0] : i * self.stride[0] + self.filtershape[0], j * self.stride[1] : j * self.stride[1] + self.filtershape[1]], axis=(1, 2))
    elif self.opperation == 'Min':
      for i in range(self.out.shape[1]):
        for j in range(self.out.shape[2]):
          self.out[:, i, j] = np.amin(input[:, i * self.stride[0] : i * self.stride[0] + self.filtershape[0], j * self.stride[1] : j * self.stride[1] + self.filtershape[1]], axis=(1, 2))
    elif self.opperation == 'Mean':
      for i in range(self.out.shape[1]):
        for j in range(self.out.shape[2]):
          self.out[:, i, j] = np.mean(input[:, i * self.stride[0] : i * self.stride[0] + self.filtershape[0], j * self.stride[1] : j * self.stride[1] + self.filtershape[1]], axis=(1, 2))
    elif self.opperation == 'Sum':
      for i in range(self.out.shape[1]):
        for j in range(self.out.shape[2]):
          self.out[:, i, j] = np.sum(input[:, i * self.stride[0] : i * self.stride[0] + self.filtershape[0], j * self.stride[1] : j * self.stride[1] + self.filtershape[1]], axis=(1, 2))
    return self.out

  def backward(self, outError):
    if self.opperation == 'Max' or self.opperation == 'Min':
      return np.repeat(np.repeat(outError, 2, axis=1), 2, axis=2) * np.equal(np.repeat(np.repeat(self.out, 2, axis=1), 2, axis=2), self.input).astype(int)
    elif self.opperation == 'Mean' or self.opperation == 'Sum':
      return np.repeat(np.repeat(outError, 2, axis=1), 2, axis=2) * np.repeat(np.repeat(self.out, 2, axis=1), 2, axis=2)