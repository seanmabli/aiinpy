import numpy as np
from emnist import extract_training_samples, extract_test_samples
from NN import NN
from CONV import CONV
from alive_progress import alive_bar

def Discriminator():
	SetSlopeForLeakyReLU(0.2)
	DisInputToCONV1 = CONV((64, 3, 3), Activation='LeakyReLU', Padding='Same', Stride=(2, 2))
	DisCONV1ToCONV2 = CONV((64, 3, 3), Activation='LeakyReLU', Padding='Same', Stride=(2, 2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	return model