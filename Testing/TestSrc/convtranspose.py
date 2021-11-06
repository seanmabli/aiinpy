import numpy as np
from .activation import *

class convtranspose:
  def __init__(self, InShape, FilterShape, LearningRate, Activation='None', Padding=False, Stride=(1, 1)):
