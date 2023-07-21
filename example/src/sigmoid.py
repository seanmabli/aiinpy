from .tensor import tensor

class sigmoid:
  def __repr__(self):
    return 'sigmoid()'

  def forward(self, input):
    return 1 / (1 + tensor.exp(-input))
    
  def backward(self, input):
    return tensor.exp(-input) / ((1 + tensor.exp(-input)) ** 2)