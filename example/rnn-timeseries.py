import src as ai
from alive_progress import alive_bar
import numpy as np

model = ai.rnn(inshape=1, outshape=1, Type='ManyToMany', outactivation=ai.identity(), learningrate=0.01)

Data = np.genfromtxt("testing\data\Timeseries\Airpassenger.csv", dtype=int)
Data = (Data - min(Data)) / (max(Data) - min(Data)).astype(float)

TrainingData = Data[0 : 100, np.newaxis]
TestData = Data[100 :, np.newaxis]

NumOfTrainGen = 15000
NumOfTestGen = len(TestData) - 5

with alive_bar(NumOfTrainGen + NumOfTestGen) as bar:
  for Generation in range(NumOfTrainGen):
    Random = np.random.randint(0, len(TrainingData) - 5)

    input = TrainingData[Random : Random + 5]
    Out = model.forward(input)

    OutError = TrainingData[Random + 1 : Random + 6] - Out
    inError = model.backward(OutError)

    bar()

  error = 0
  for Generation in range(NumOfTestGen):
    input = TestData[Generation : Generation + 5]
    Out = model.forward(input)

    OutError = TestData[Generation + 1 : Generation + 6] - Out
    error += np.sum(OutError)
    inError = model.backward(OutError)

    bar()

  print(error / NumOfTestGen)