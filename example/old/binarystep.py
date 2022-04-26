import numpy as np

class binarystep:
  def forward(self, input):
    equation = np.vectorize(self.equationForbinaryStep, otypes=[float])
    return equation(input)

  def equationForbinaryStep(self, input):
    return 0 if (input < 0) else 1
    
  def backward(self, input):
    return 1