import numpy as np
from GRU import GRU

with open('example\GRU\Data\English.txt') as Data:
  EnglishData = Data.readlines()
with open('example\GRU\Data\French.txt') as Data:
  FrenchData = Data.readlines()

DataUniqueWords = list(set([w for Sentence in EnglishData + FrenchData for w in Sentence.split(' ')]))

'''
GRU = GRU(len(TrainingDataUniqueWords), 2, LearningRate=0.05)

for Generation in range(15000):
  items = list(TrainingData.items())
  Random = np.random.randint(0, len(TrainingDataUniqueWords))

  InputSentenceSplit = list(items[Random][0].split(' '))
  Input = np.zeros((len(InputSentenceSplit), len(TrainingDataUniqueWords)))
  for i in range(len(InputSentenceSplit)):
    Input[i, TrainingDataUniqueWords.index(InputSentenceSplit[i])] = 1
'''