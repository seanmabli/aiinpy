class identity:
  def __repr__(self):
    return 'identity()'

  def forward(self, input):
    return input
    
  def backward(self, input):
    return 1