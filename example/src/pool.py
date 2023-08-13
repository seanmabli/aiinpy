from .tensor import tensor

class pool: 
  def __init__(self, stride, filtershape, operation, inshape=None):
    self.stride, self.filtershape, self.operation, self.inshape = stride, filtershape, operation, inshape
    if inshape is not None and len(inshape) == 2:
      outshape = tuple([int(tensor.floor(inshape[0] / stride[0])), int(tensor.floor(inshape[1] / stride[1]))])
      while outshape[0] * stride[0] + filtershape[0] - stride[0] < inshape[0]:
        outshape[0] -= 1
      while outshape[1] * stride[1] + filtershape[1] - stride[1] < inshape[1]:
        outshape[1] -= 1
      self.outshape, self.out = outshape, tensor.zeros(outshape)
    elif inshape is not None and len(inshape) == 3:
      outshape = tuple([inshape[0], int(tensor.floor(inshape[1] / stride[0])), int(tensor.floor(inshape[2] / stride[1]))])
      while outshape[1] * stride[0] + filtershape[0] - stride[0] < inshape[1]:
        outshape[1] -= 1
      while outshape[2] * stride[1] + filtershape[1] - stride[1] < inshape[2]:
        outshape[2] -= 1
      self.outshape, self.out = outshape, tensor.zeros(outshape)

  def __copy__(self):
    return type(self)(self.stride, self.filtershape, self.operation, self.inshape)

  def __repr__(self):
    return 'pool(inshape=' + str(self.inshape) + ', outshape=' + str(self.outshape) + ', stride=' + str(self.stride) + ', filtershape=' + str(self.filtershape) + ', operation=' + self.operation + ')'

  def modelinit(self, inshape):
    self.inshape = inshape
    if len(inshape) == 2:
      outshape = tuple([int(tensor.floor(inshape[0] / self.stride[0])), int(tensor.floor(inshape[1] / self.stride[1]))])
      while outshape[0] * self.stride[0] + self.filtershape[0] - self.stride[0] < inshape[0]:
        outshape[0] -= 1
      while outshape[1] * self.stride[1] + self.filtershape[1] - self.stride[1] < inshape[1]:
        outshape[1] -= 1
    elif len(inshape) == 3:
      outshape = tuple([inshape[0], int(tensor.floor(inshape[1] / self.stride[0])), int(tensor.floor(inshape[2] / self.stride[1]))])
      while outshape[1] * self.stride[0] + self.filtershape[0] - self.stride[0] < inshape[1]:
        outshape[1] -= 1
      while outshape[2] * self.stride[1] + self.filtershape[1] - self.stride[1] < inshape[2]:
        outshape[2] -= 1
    self.outshape, self.out = outshape, tensor.zeros(outshape)
    return outshape

  def forward(self, input):
    self.input = input

    if self.operation == 'max':
      for i in range(self.outshape[1]):
        for j in range(self.outshape[2]):
          self.out[:, i, j] = tensor.max(input[:, i * self.stride[0] : i * self.stride[0] + self.filtershape[0], j * self.stride[1] : j * self.stride[1] + self.filtershape[1]], axis=(1, 2))
    elif self.operation == 'min':
      for i in range(self.outshape[1]):
        for j in range(self.outshape[2]):
          self.out[:, i, j] = tensor.min(input[:, i * self.stride[0] : i * self.stride[0] + self.filtershape[0], j * self.stride[1] : j * self.stride[1] + self.filtershape[1]], axis=(1, 2))
    elif self.operation == 'mean':
      for i in range(self.outshape[1]):
        for j in range(self.outshape[2]):
          self.out[:, i, j] = tensor.mean(input[:, i * self.stride[0] : i * self.stride[0] + self.filtershape[0], j * self.stride[1] : j * self.stride[1] + self.filtershape[1]], axis=(1, 2))
    elif self.operation == 'sum':
      for i in range(self.outshape[1]):
        for j in range(self.outshape[2]):
          self.out[:, i, j] = tensor.sum(input[:, i * self.stride[0] : i * self.stride[0] + self.filtershape[0], j * self.stride[1] : j * self.stride[1] + self.filtershape[1]], axis=(1, 2))
    return self.out

  def backward(self, outError):
    if self.operation == 'max' or self.operation == 'min':
      return tensor.repeat(tensor.repeat(outError, 2, axis=1), 2, axis=2) * (tensor.repeat(tensor.repeat(self.out, 2, axis=1), 2, axis=2) == self.input).astype(int)
    elif self.operation == 'mean' or self.operation == 'sum':
      return tensor.repeat(tensor.repeat(outError, 2, axis=1), 2, axis=2) * tensor.repeat(tensor.repeat(self.out, 2, axis=1), 2, axis=2)