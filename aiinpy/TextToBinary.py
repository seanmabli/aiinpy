def TextToBinary(In):
  BinaryInput = np.zeros((len(WordToBinary(In)), 8))
  for i in range(len(WordToBinary(Input))):
    BinaryInput[i, :] = np.array(list(WordToBinary(In)[i]))
  return BinaryInput