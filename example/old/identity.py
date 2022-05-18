class identity:
  def forward(self, input):
    return input
    
  def backward(self, input):
    return 1