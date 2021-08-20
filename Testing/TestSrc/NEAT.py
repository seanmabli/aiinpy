import numpy as np

class NEAT:
  def __init__(self, InSize, OutSize):
    NodeGenes = np.zeros((InSize + OutSize)) # Global (Inovation Number)
    ConnectGenes = np.zeros(5) # Global, (In Node, Out Node, Weight, Enabled / Disabled, Innovation Number) * Num Of Nodes

  def ForwardProp(self, In):
    print("Forward Prop")

  def Mutate(self):
    print("Mutate")

  def Crossover(self):
    print("Crossover")

  def Speciation(self):
    print("Speciation") 