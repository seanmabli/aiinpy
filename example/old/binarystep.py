import numpy as np

class binarystep:
  def __repr__(self):
    return 'binarystep()'

  def forward(self, input):
    equation = np.vectorize(self.equationForbinaryStep, otypes=[float])
    return equation(input)

  def equationForbinaryStep(self, input):
    return 0 if (input < 0) else 1
    
  def backward(self, input):
    return 0
    # if input == 0 return undefined, this was removed because it would return an error 