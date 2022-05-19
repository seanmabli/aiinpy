import src as ai
import numpy as np
import wandb

wandb.init(project='lstm-timeseries', config={'version' : 'new-a'})

model = ai.lstm(inshape=1, outshape=1, outactivation=ai.identity(), learningrate=0.01)

Data = np.genfromtxt(r"data/timeseries/airpassenger.csv", dtype=int)
Data = (Data - min(Data)) / (max(Data) - min(Data)).astype(float)

TrainingData = Data[0 : 100, np.newaxis]
TestData = Data[100 :, np.newaxis]

NumOfTrainGen = 15000
NumOfTestGen = len(TestData) - 5

for Generation in range(NumOfTrainGen):
  Random = np.random.randint(0, len(TrainingData) - 5)

  input = TrainingData[Random : Random + 5]
  Out = model.forward(input)

  OutError = TrainingData[Random + 1 : Random + 6] - Out
  inError = model.backward(OutError)

error = 0
for Generation in range(NumOfTestGen):
  input = TestData[Generation : Generation + 5]
  Out = model.forward(input)

  OutError = TestData[Generation + 1 : Generation + 6] - Out
  error += np.sum(OutError)
  inError = model.backward(OutError)

print(error / NumOfTestGen)
wandb.log({"test accuracy" : error / NumOfTestGen})