from testsrc.lstm import lstm
from alive_progress import alive_bar
import numpy as np
import wandb
import sys

wandb.init(project='lstm')

lstm_model = lstm(InSize=1, OutSize=1, LearningRate=0.01)

Data = np.genfromtxt('testing\data\Timeseries\Airpassenger.csv', dtype=int)
Data = (Data - min(Data)) / (max(Data) - min(Data)).astype(float)

TrainingData = Data[0 : 100, np.newaxis]
TestData = Data[100 :, np.newaxis]

NumOfTrainGen = 15000
NumOfTestGen = len(TestData)

with alive_bar(NumOfTrainGen) as bar:
  for Generation in range(NumOfTrainGen):
    Random = np.random.randint(0, len(TrainingData) - 5)

    In = TrainingData[Random : Random + 5]
    Out = lstm_model.forwardprop(In)

    OutError = TrainingData[Random + 1 : Random + 6] - Out
    InError = lstm_model.backprop(OutError)

    wandb.log({'Out Error': np.sum(abs(OutError))})
    bar()