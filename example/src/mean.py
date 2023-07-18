from .tensor import tensor

class mean:
  def __init__(self, axis=0):
      self.axis = axis

  def __repr__(self):
    return 'average()'

  def modelinit(self, inshape):
    self.inshape = inshape
    return inshape[:self.axis] + inshape[self.axis + 1:]
    
  def forward(self, input):
    return input.mean(axis=self.axis)

  def backward(self, input): # fix
    # return tensor(np.ones(self.inshape) / self.inshape[self.axis])
    # return np.array([input] * self.inshape[self.axis])
    return input # fix
