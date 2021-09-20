import numpy as np
import pandas as pd
from TestSrc.LSTM import LSTM
# import wandb

# wandb.init(project='lstm')

LSTM_Model = LSTM(InSize=1, OutSize=1, LearningRate=0.01)

Data = np.genfromtxt('Testing\Data\TimeSeriesData\AirPassengers.csv', dtype=int)
Data = (Data - min(Data)) / (max(Data) - min(Data)).astype(float)

TrainingData = Data[0 : 100, np.newaxis]
TestData = Data[100 :, np.newaxis]

Out = LSTM_Model.ForwardProp(TrainingData)
OutError = 
print(Out)