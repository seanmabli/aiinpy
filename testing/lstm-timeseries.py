import testsrc as ai
from alive_progress import alive_bar
import numpy as np
import wandb

lstm_model = ai.lstm(InSize=1, OutSize=1, OutActivation=ai.identity(), LearningRate=0.01)

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

    bar()

  for Generation in range(NumOfTestGen):
    In = TestData[Generation : Generation + 5]
    Out = lstm_model.forward(In)

    OutError = TestData[Generation + 1 : Generation + 6] - Out
    InError = lstm_model.backward(OutError)

    bar()