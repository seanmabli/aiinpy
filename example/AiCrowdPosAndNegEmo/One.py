from RNN import RNN
import pandas as pd
import numpy as np
from alive_progress import alive_bar

TrainingData = pd.read_csv("data/train.csv")[0:10000].to_numpy()
# ValidationData = pd.read_csv("data/val.csv")[1:].to_numpy()
TestData = pd.read_csv("data/test.csv")[0:1000].to_numpy()

TrainingDataUniqueWords = list(set([w for Sentence in TrainingData[:, 0] for w in Sentence]))
Rnn = RNN(len(TrainingDataUniqueWords), 2, LearningRate=0.05)

print(list(TrainingData[0, 0]))

NumOfTrainGen = 15000
NumOfTestGen = 1000

with alive_bar(NumOfTrainGen + NumOfTestGen) as bar:
  for Generation in range(NumOfTrainGen):
    Random = np.random.randint(0, len(TrainingData))

    InputSentenceSplit = list(TrainingData[Random, 0])
    Input = np.zeros((len(InputSentenceSplit), len(TrainingDataUniqueWords)))
    for i in range(len(InputSentenceSplit)):
      Input[i, TrainingDataUniqueWords.index(InputSentenceSplit[i])] = 1

    Output = Rnn.ForwardProp(Input)

    RealOutput = np.zeros(Output.shape)
    RealOutput[(1 if TrainingData[Random, 1] == True else 0)] = 1

    NumberCorrect = int(np.argmax(Output) == (1 if TrainingData[Random, 1] == True else 0))

    OutputError = RealOutput - Output
    Rnn.BackProp(OutputError)
    bar()

  NumberCorrect = 0
  for Generation in range(NumOfTestGen):
    InputSentenceSplit = list(TestData[Generation, 0])
    Input = np.zeros((len(InputSentenceSplit), len(TrainingDataUniqueWords)))
    for i in range(len(InputSentenceSplit)):
      Input[i, TrainingDataUniqueWords.index(InputSentenceSplit[i])] = 1

    Output = Rnn.ForwardProp(Input)
    NumberCorrect += int(np.argmax(Output) == (1 if TestData[Generation, 1] == True else 0))
    bar()

  print(NumberCorrect / NumOfTestGen)
