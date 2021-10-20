from TestSrc.LSTM import LSTM
from alive_progress import alive_bar
import numpy as np
import pandas as pd
import wandb

wandb.init(project='lstm')

LSTM_Model = LSTM(InSize=1, OutSize=1, LearningRate=0.01)

Data = np.genfromtxt('Testing\Data\TimeSeriesData\AirPassengers.csv', dtype=int)
Data = (Data - min(Data)) / (max(Data) - min(Data)).astype(float)

TrainingData = Data[0 : 100, np.newaxis]
TestData = Data[100 :, np.newaxis]

NumOfTrainGen = 15000
NumOfTestGen = len(TestData)

with alive_bar(NumOfTrainGen) as bar:
  for Generation in range(NumOfTrainGen):
    Random = np.random.randint(0, len(TrainingData) - 5)

    In = TrainingData[Random : Random + 5]
    Out = LSTM_Model.ForwardProp(In)

    OutError = Out - TrainingData[Random + 1 : Random + 6]
    InError = LSTM_Model.BackProp(OutError)

    wandb.log({'Out Error': abs(np.sum(OutError))})
    bar()