import numpy as np

class average:
  def __init__(self, axis=0):
      self.axis = axis

  def __repr__(self):
    return 'average()'

  def modelinit(self, inshape):
    self.inshape = inshape
    return inshape[:self.axis] + inshape[self.axis + 1:]
    
  def forward(self, input):
    return np.average(input, axis=self.axis)

  def backward(self, input):
    return np.array([input] * self.inshape[self.axis])