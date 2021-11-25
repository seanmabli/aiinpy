import numpy as np

class NEaT:
  def __init__(self, inshape, outshape):
    NodeGenes = np.zeros((inshape + outshape)) # Global (inovation Number)
    connectGenes = np.zeros(5) # Global, (in Node, out Node, Weight, Enabled / Disabled, innovation Number) * Num Of Nodes

  def forward(self, in):
    print("Forward Prop")

  def Mutate(self):
    print("Mutate")

  def crossover(self):
    print("crossover")

  def Speciation(self):
    print("Speciation") 