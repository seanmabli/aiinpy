class texttobinary:
  def forward(In):
    Dec = list(bytearray(In, "utf8"))
    Out = [''] * len(In)
    for i in range(len(In)):
      Out[i] = bin(Dec[i]).replace("b", ("0"*(9-len(bin(Dec[i])))))
    return Out
  def backward(In):
    return In

class binarytotext:
  def forward(In):
    Out = ''
    for i in range(len(In)):
      Out += chr(int(In[i], 2))
    return Out