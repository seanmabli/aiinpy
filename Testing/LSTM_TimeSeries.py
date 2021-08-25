import numpy as np
import pandas as pd
from TestSrc.LSTM import LSTM
# import wandb

# wandb.init(project='lstm')

LSTM_Model = LSTM(InSize=6, OutSize=3, LearningRate=0.01)

Data = np.genfromtxt('Testing\Data\TimeSeriesData\AirPassengers.csv', dtype=int)

TrainingData = Data[0 : 100]
TestData = Data[100 :]

print(TrainingData)
print(TestData)

'''
Out = LSTM_Model.ForwardProp(TraingingData[0:5, :6].astype(float) / 10)
OutError = TraingingData[0:5, 6:9] - Out
LSTM_Model.BackProp(OutError)
'''