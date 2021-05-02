import numpy as np
import aiinpy as ai

PositiveComments = open("C:\\Users\\smdro\\Downloads\\archive\\sentence_polarity\\rt-polarity.pos", "r")
NegativeComments = open("C:\\Users\\smdro\\Downloads\\archive\\sentence_polarity\\rt-polarity.neg", "r")
Comments = str(PositiveComments.read() + NegativeComments.read())
Comments = np.array(Comments.splitlines())

TrainingDataUniqueWords = list(set([w for Sentence in Comments for w in Sentence.split(' ')]))
Rnn = ai.RNN(len(TrainingDataUniqueWords), 2, "ManyToOne")