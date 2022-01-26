import numpy as np

class relu:
  def forward(self, input):
    output = np.maximum(0, input)
    return output

  def backward(self, input):
    equation = np.vectorize(self.equationforderivative, otypes=[float])
    return equation(input)
    
  def equationforderivative(self, input):
    if input < 0:
      return 0
    elif input == 0:
      return None
    elif input > 0:    
      return 1
