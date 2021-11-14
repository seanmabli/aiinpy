from testsrc.lstm import lstm
from alive_progress import alive_bar
import numpy as np
import wandb
import sys

wandb.init(project='lstm')

lstm_model = lstm(InSize=1, OutSize=1, OutActivation='Identity', LearningRate=0.01)

Data = np.genfromtxt('testing\data\Timeseries\Airpassenger.csv', dtype=int)
Data = (Data - min(Data)) / (max(Data) - min(Data)).astype(float)

TrainingData = Data[0 : 100, np.newaxis]
TestData = Data[100 :, np.newaxis]

NumOfTrainGen = 15000
NumOfTestGen = len(TestData) - 6

with alive_bar(NumOfTrainGen + NumOfTestGen) as bar:
  for Generation in range(NumOfTrainGen):
    Random = np.random.randint(0, len(TrainingData) - 5)

    In = TrainingData[Random : Random + 5]  
    Out = lstm_model.forward(In)
    
    OutError = TrainingData[Random + 1 : Random + 6] - Out
    InError = lstm_model.backward(OutError)

    wandb.log({'Out Error': np.sum(abs(OutError))})
    bar()

  for Generation in range(NumOfTestGen):
    In = TestData[Generation : Generation + 5]
    Out = lstm_model.forward(In)

    OutError = TestData[Generation + 1 : Generation + 6] - Out
    InError = lstm_model.backward(OutError)

    bar()

    wandb.log({'Real Data': TestData[Generation + 6], 'Prediction': Out[4], 'Prediction Accuracy': abs(TestData[Generation + 6] - Out[4])})