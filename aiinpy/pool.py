import numpy as np
from numpy.lib.stride_tricks import as_strided

class pool: 
  def __init__(self, stride, filtershape, opperation, inshape=None):
    self.stride, self.filtershape, self.opperation, self.inshape = stride, filtershape, opperation, inshape
    if inshape is not None:
      self._computeoutshape(inshape)

  def _computeoutshape(self, inshape):
    if len(inshape) == 2:
      oh = (inshape[0] - self.filtershape[0]) // self.stride[0] + 1
      ow = (inshape[1] - self.filtershape[1]) // self.stride[1] + 1
      self.outshape = (oh, ow)
    elif len(inshape) == 3:
      oh = (inshape[1] - self.filtershape[0]) // self.stride[0] + 1
      ow = (inshape[2] - self.filtershape[1]) // self.stride[1] + 1
      self.outshape = (inshape[0], oh, ow)
    self.out = np.zeros(self.outshape)

  def __copy__(self):
    return type(self)(self.stride, self.filtershape, self.opperation, self.inshape)

  def modelinit(self, inshape):
    self.inshape = inshape
    self._computeoutshape(inshape)
    return self.outshape

  def forward(self, input):
    self.input = input

    c = self.outshape[0]
    oh, ow = self.outshape[1], self.outshape[2]
    fh, fw = self.filtershape
    sh, sw = self.stride

    s0, s1, s2 = input.strides
    windowed = as_strided(input,
      shape=(c, oh, ow, fh, fw),
      strides=(s0, s1 * sh, s2 * sw, s1, s2))

    if self.opperation == 'Max':
      self.out = windowed.max(axis=(3, 4))
    elif self.opperation == 'Min':
      self.out = windowed.min(axis=(3, 4))
    elif self.opperation == 'Mean':
      self.out = windowed.mean(axis=(3, 4))
    elif self.opperation == 'Sum':
      self.out = windowed.sum(axis=(3, 4))
    return self.out

  def backward(self, outError):
    if self.opperation == 'Max' or self.opperation == 'Min':
      return np.repeat(np.repeat(outError, 2, axis=1), 2, axis=2) * np.equal(np.repeat(np.repeat(self.out, 2, axis=1), 2, axis=2), self.input).astype(int)
    elif self.opperation == 'Mean' or self.opperation == 'Sum':
      return np.repeat(np.repeat(outError, 2, axis=1), 2, axis=2) * np.repeat(np.repeat(self.out, 2, axis=1), 2, axis=2)
