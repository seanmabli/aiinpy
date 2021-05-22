import numpy as np
from RNN import RNN
from Data.PosNegCon import TrainingData, TestData

TrainingDataUniqueWords = list(set([w for Sentence in TrainingData.keys() for w in Sentence.split(' ')]))
Rnn = RNN(len(TrainingDataUniqueWords), 2, Type='ManyToOne', LearningRate=0.05)

for Generation in range(15000):
  items = list(TrainingData.items())
  Random = np.random.randint(0, len(TrainingDataUniqueWords))

  InputSentenceSplit = list(items[Random][0].split(' '))
  Input = np.zeros((len(InputSentenceSplit), len(TrainingDataUniqueWords)))
  for i in range(len(InputSentenceSplit)):
    Input[i, TrainingDataUniqueWords.index(InputSentenceSplit[i])] = 1

  Output = Rnn.ForwardProp(Input)

  RealOutput = np.zeros(Output.shape)
  RealOutput[(1 if items[Random][1] == True else 0)] = 1

  NumberCorrect = int(np.argmax(Output) == (1 if items[Random][1] == True else 0))

  OutputError = RealOutput - Output
  Rnn.BackProp(OutputError)

items = list(TestData.items())
np.random.shuffle(items)

NumberCorrect = 0

for x, y in items:
  InputSentenceSplit = list(x.split(' '))
  Input = np.zeros((len(InputSentenceSplit), len(TrainingDataUniqueWords)))
  for i in range(len(InputSentenceSplit)):
    Input[i, TrainingDataUniqueWords.index(InputSentenceSplit[i])] = 1

  Output = Rnn.ForwardProp(Input)
  NumberCorrect += int(np.argmax(Output) == (1 if y == True else 0))
  
print(NumberCorrect / len(TestData))