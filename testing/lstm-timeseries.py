import testsrc as ai
from alive_progress import alive_bar
import numpy as np
import wandb

lstm_model = ai.lstm(inshape=1, outshape=1, outactivation=ai.identity(), learningrate=0.01)

Data = np.genfromtxt('testing\data\Timeseries\airpassenger.csv', dtype=int)
Data = (Data - min(Data)) / (max(Data) - min(Data)).astype(float)

TrainingData = Data[0 : 100, np.newaxis]
TestData = Data[100 :, np.newaxis]

NumOfTrainGen = 15000
NumOfTestGen = len(TestData) - 6

with alive_bar(NumOfTrainGen + NumOfTestGen) as bar:
  for Generation in range(NumOfTrainGen):
    Random = np.random.randint(0, len(TrainingData) - 5)

    in = TrainingData[Random : Random + 5]  
    out = lstm_model.forward(in)
    
    outError = TrainingData[Random + 1 : Random + 6] - out
    inError = lstm_model.backward(outError)

    bar()

  for Generation in range(NumOfTestGen):
    in = TestData[Generation : Generation + 5]
    out = lstm_model.forward(in)

    outError = TestData[Generation + 1 : Generation + 6] - out
    inError = lstm_model.backward(outError)

    bar()