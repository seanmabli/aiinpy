import numpy as np

class average:
  def __init__(self, axis=0):
      self.axis = axis

  def __repr__(self):
    return 'average()'

  def modelinit(inshape):
    self.inshape = inshape
    print(inshape, inshape[:axis] + inshape[axis:])

  def forward(self, input):
    return np.average(input, axis=self.axis)

  def backward(self, input):
    return np.array([input] * inshape[self.axis])