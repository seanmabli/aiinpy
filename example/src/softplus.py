class softplus:
  def forward(self, input):
    return np.log(1 + np.exp(input))
    
  def backward(self, input):
    return 1 / (1 + np.exp(-input))