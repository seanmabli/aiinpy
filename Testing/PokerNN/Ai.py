from NN import NN
import numpy as np

class PokerModel:
  def __init__(self):
    # Preflop NN:
    InToHid1 = NN(InSize=7, OutSize=32, Activation='Sigmoid', LearingRate=0)
    Hid1ToHid2 = NN(InSize=32, OutSize=32, Activation='Sigmoid', LearingRate=0)
    Hid2ToHid3 = NN(InSize=32, OutSize=32, Activation='Sigmoid', LearingRate=0)
    Hid3ToHid4 = NN(InSize=32, OutSize=32, Activation='Sigmoid', LearingRate=0)
    Hid4ToOut = NN(InSize=32, OutSize=9, Activation='Sigmoid', LearingRate=0)

    # Flop NN:
    InToHid1 = NN(InSize=8, OutSize=32, Activation='Sigmoid', LearingRate=0)
    Hid1ToHid2 = NN(InSize=32, OutSize=32, Activation='Sigmoid', LearingRate=0)
    Hid2ToHid3 = NN(InSize=32, OutSize=32, Activation='Sigmoid', LearingRate=0)
    Hid3ToHid4 = NN(InSize=32, OutSize=32, Activation='Sigmoid', LearingRate=0)
    Hid4ToOut = NN(InSize=32, OutSize=9, Activation='Sigmoid', LearingRate=0)

    # Flop + Turn NN:
    InToHid1 = NN(InSize=9, OutSize=32, Activation='Sigmoid', LearingRate=0)
    Hid1ToHid2 = NN(InSize=32, OutSize=32, Activation='Sigmoid', LearingRate=0)
    Hid2ToHid3 = NN(InSize=32, OutSize=32, Activation='Sigmoid', LearingRate=0)
    Hid3ToHid4 = NN(InSize=32, OutSize=32, Activation='Sigmoid', LearingRate=0)
    Hid4ToOut = NN(InSize=32, OutSize=9, Activation='Sigmoid', LearingRate=0)

    # Flop + Turn + River NN:
    InToHid1 = NN(InSize=10, OutSize=32, Activation='Sigmoid', LearingRate=0)
    Hid1ToHid2 = NN(InSize=32, OutSize=32, Activation='Sigmoid', LearingRate=0)
    Hid2ToHid3 = NN(InSize=32, OutSize=32, Activation='Sigmoid', LearingRate=0)
    Hid3ToHid4 = NN(InSize=32, OutSize=32, Activation='Sigmoid', LearingRate=0)
    Hid4ToOut = NN(InSize=32, OutSize=9, Activation='Sigmoid', LearingRate=0)