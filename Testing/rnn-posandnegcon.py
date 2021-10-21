import numpy as np
from aiinpy import RNN
from alive_progress import alive_bar

import pandas as pd
TrainingData = pd.read_csv('Testing\Data\PosNegCon\Complicated\ImdbData.csv', header=None, index_col=0, squeeze=True).to_dict()[0:40000]
TestData = pd.read_csv('Testing\Data\PosNegCon\Complicated\ImdbData.csv', header=None, index_col=0, squeeze=True).to_dict()[40000:50000]

TrainingDataUniqueWords = list(set([w for Sentence in TrainingData.keys() for w in Sentence.split(' ')]))
Rnn = RNN(len(TrainingDataUniqueWords), 2, LearningRate=0.05)

NumOfTrainGen = 15000
NumOfTestGen = len(list(TestData.items()))

with alive_bar(NumOfTrainGen + NumOfTestGen) as bar:
  for Generation in range(NumOfTrainGen):
    items = list(TrainingData.items())
    Random = np.random.randint(0, len(TrainingDataUniqueWords))

    InputSentenceSplit = list(items[Random][0].split(' '))
    Input = np.zeros((len(InputSentenceSplit), len(TrainingDataUniqueWords)))
    for i in range(len(InputSentenceSplit)):
      Input[i, TrainingDataUniqueWords.index(InputSentenceSplit[i])] = 1

    Output = Rnn.forwardprop(Input)

    RealOutput = np.zeros(Output.shape)
    RealOutput[(1 if items[Random][1] == True else 0)] = 1

    NumberCorrect = int(np.argmax(Output) == (1 if items[Random][1] == True else 0))

    OutputError = RealOutput - Output
    Rnn.backprop(OutputError)
    bar()

  NumberCorrect = 0
  for Generation in range(NumOfTestGen):
    items = list(TestData.items())

    InputSentenceSplit = list(items[Generation][0].split(' '))
    Input = np.zeros((len(InputSentenceSplit), len(TrainingDataUniqueWords)))
    for i in range(len(InputSentenceSplit)):
      Input[i, TrainingDataUniqueWords.index(InputSentenceSplit[i])] = 1

    Output = Rnn.forwardprop(Input)
    NumberCorrect += int(np.argmax(Output) == (1 if items[Generation][1] == True else 0))
    bar()

  print(NumberCorrect / NumOfTestGen)