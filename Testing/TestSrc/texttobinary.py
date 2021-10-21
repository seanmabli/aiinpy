def texttobinary(In):
  Dec = list(bytearray(In, "utf8"))
  Out = [''] * len(In)
  for i in range(len(In)):
    Out[i] = bin(Dec[i]).replace("b", ("0"*(9-len(bin(Dec[i])))))
  return Out