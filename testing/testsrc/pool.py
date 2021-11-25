import numpy as np

class pool: 
  def __init__(self, Stride, Filtershape, Type):
    self.Stride, self.Filtershape, self.Type = Stride, Filtershape, Type
    
  def __copy__(self):
    return type(self)(self.Stride, self.Filtershape, self.Type)

  def forward(self, input):
    self.input = input
    self.out = np.zeros((len(input), int(np.floor(len(input[0]) / self.Stride[0])), int(np.floor(len(input[0, 0]) / self.Stride[1]))))
    while self.out.shape[1] * self.Stride[0] + self.Filtershape[0] - self.Stride[0] < input.shape[1]:
      self.out = self.out[:, : - 1, :]
    while self.out.shape[2] * self.Stride[1] + self.Filtershape[1] - self.Stride[1] < input.shape[2]:
      self.out = self.out[:, :, : - 1]

    if self.Type == 'Max':
      for i in range(self.out.shape[1]):
        for j in range(self.out.shape[2]):
          self.out[:, i, j] = np.amax(input[:, i * self.Stride[0] : i * self.Stride[0] + 2, j * self.Stride[1] : j * self.Stride[1] + 2], axis=(1, 2))
    elif self.Type == 'Min':
      for i in range(self.out.shape[1]):
        for j in range(self.out.shape[2]):
          self.out[:, i, j] = np.amin(input[:, i * self.Stride[0] : i * self.Stride[0] + 2, j * self.Stride[1] : j * self.Stride[1] + 2], axis=(1, 2))
    elif self.Type == 'Mean':
      for i in range(self.out.shape[1]):
        for j in range(self.out.shape[2]):
          self.out[:, i, j] = np.mean(input[:, i * self.Stride[0] : i * self.Stride[0] + 2, j * self.Stride[1] : j * self.Stride[1] + 2], axis=(1, 2))
    elif self.Type == 'Sum':
      for i in range(self.out.shape[1]):
        for j in range(self.out.shape[2]):
          self.out[:, i, j] = np.sum(input[:, i * self.Stride[0] : i * self.Stride[0] + 2, j * self.Stride[1] : j * self.Stride[1] + 2], axis=(1, 2))
    return self.out

  def backward(self, outError):
    if self.Type == 'Max' or self.Type == 'Min':
      return np.repeat(np.repeat(outError, 2, axis=1), 2, axis=2) * np.equal(np.repeat(np.repeat(self.out, 2, axis=1), 2, axis=2), self.input).astype(int)
    elif self.Type == 'Mean' or self.Type == 'Sum':
      return np.repeat(np.repeat(outError, 2, axis=1), 2, axis=2) * np.repeat(np.repeat(self.out, 2, axis=1), 2, axis=2)