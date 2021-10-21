from RNN_ManyToMany import RNN
from alive_progress import alive_bar
import numpy as np
import pandas as pd
import wandb

wandb.init(project='rnn-timeseries')

RNN_Model = RNN(InSize=1, OutSize=1, LearningRate=0.01)

Data = np.genfromtxt('Testing\Data\TimeSeriesData\AirPassengers.csv', dtype=int)
Data = (Data - min(Data)) / (max(Data) - min(Data)).astype(float)

TrainingData = Data[0 : 100, np.newaxis]
TestData = Data[100 :, np.newaxis]

NumOfTrainGen = 15000
NumOfTestGen = len(TestData)

with alive_bar(NumOfTrainGen + NumOfTestGen) as bar:
  for Generation in range(NumOfTrainGen):

Out = RNN_Model.forwardprop(TrainingData)
OutError = 
print(Out)