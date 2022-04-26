class texttobinary:
  def forward(input):
    Dec = list(bytearray(input, "utf8"))
    out = [''] * len(input)
    for i in range(len(input)):
      out[i] = bin(Dec[i]).replace("b", ("0"*(9-len(bin(Dec[i])))))
    return out
    
  def backward(input):
    return input