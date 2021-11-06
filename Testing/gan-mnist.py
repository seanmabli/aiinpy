import numpy as np
from emnist import extract_training_samples, extract_test_samples
from testsrc.model import model
from testsrc.conv import conv
from testsrc.convtranspose import convtranspose
from testsrc.nn import nn

# Dis -> Discrimanator
dis_model = model((28, 28), 1, [
  conv((28, 28), (64, 3, 3), 0.0002, 'LeakyReLU', True, (2, 2)),
  conv((64, 14, 14), (64, 3, 3), 0.0002, 'LeakyReLU', True, (2, 2)),
  nn((64, 7, 7), 1,'Sigmoid', 0.0002)
])

dis_model.Model[0].SetSlopeForLeakyReLU(0.2)
dis_model.Model[1].SetSlopeForLeakyReLU(0.2)

gen_model = model(10, (28, 28), [
  nn(10, (128, 7, 7), 0.0002, 'LeakyReLU'),


# Train Discrimanator
DisRealData, _ = extract_training_samples('digits')
DisRealData = (DisRealData[0 : 100000] / 255) - 0.5
DisFakeData = np.random.uniform(-0.5, 0.5, (len(DisRealData), 28, 28))
TrainDataIn = np.vstack((DisRealData, DisFakeData))
TrainDataOut = np.hstack((np.ones(len(DisRealData)), np.zeros(len(DisFakeData))))

dis_model.train(TrainDataIn, TrainDataOut, 2000)
print(dis_model.test(TrainDataIn[:50000], TrainDataOut[:50000]))