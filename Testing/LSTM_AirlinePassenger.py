import numpy as np
import pandas as pd
from TestSrc.LSTM import LSTM

LSTM_Model = LSTM(InSize=6, OutSize=3, LearningRate=0.01)

x = np.genfromtxt('Testing\Data\TimeSeriesData\AirlinePassengers.csv', dtype=str)

Data = np.array(([[''] * 13] * len(x)))
for i in range(len(x)):
  Data[i, :] = list(x[i])

# Data, Year 1 - 4, Month 6 - 7, Passengers 10 - 12
Data = np.delete(Data, [0, 5, 8, 9], 1)
# In 0 - 6, Out 7 - 9
