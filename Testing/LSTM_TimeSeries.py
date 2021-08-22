import numpy as np
import pandas as pd
from TestSrc.LSTM import LSTM

LSTM_Model = LSTM(InSize=6, OutSize=3, LearningRate=0.01)

x = np.genfromtxt('Testing\Data\TimeSeriesData\AirlinePassengers.csv', dtype=str)

Data = np.array(([[''] * 13] * len(x)))
for i in range(len(x)):
  Data[i, :] = list(x[i])

TraingingData = np.delete(Data, [0, 5, 8, 9], 1)[0:100, :].astype(float) / 10
TestData = np.delete(Data, [0, 5, 8, 9], 1)[100:, :].astype(float) / 10

Out = LSTM_Model.ForwardProp(TraingingData[0:5, :6].astype(float) / 10)
OutError = TraingingData[0:5, 6:9] - Out
LSTM_Model.BackProp(OutError)