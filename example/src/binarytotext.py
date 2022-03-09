class binarytotext:
  def forward(input):
    out = ''
    for i in range(len(input)):
      out += chr(int(input[i], 2))
    return out
    
  def backward(input):
    return input