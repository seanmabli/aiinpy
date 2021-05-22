import numpy as np
from emnist import extract_training_samples, extract_test_samples
from NN import NN
from CONV import CONV
from alive_progress import alive_bar
import time

# Dis -> Discrimanator
DisInputToCONV1 = CONV((64, 3, 3), LearningRate=0.0002, Activation='LeakyReLU', Padding='Same', Stride=(2, 2), DropoutRate=0)
DisCONV1ToCONV2 = CONV((64, 3, 3), LearningRate=0.0002, Activation='LeakyReLU', Padding='Same', Stride=(2, 2), DropoutRate=0)
DisInputToOutput = NN(InputSize=(64 * 7 * 7), OutputSize=1, LearningRate=0.0002, Activation='Sigmoid')
DisInputToCONV1.SetSlopeForLeakyReLU(0.2)
DisCONV1ToCONV2.SetSlopeForLeakyReLU(0.2)

# Train Discrimanator
DisRealData, _ = extract_training_samples('digits')
DisRealData = DisRealData[:100000]
DisFakeData = np.random.uniform(-0.5, 0.5, (len(DisRealData), 28, 28))
Data = np.vstack((DisRealData, DisFakeData))
TotalError = 0

for Generation in range(1):
  Random = np.random.randint(0, len(Data))
  RealOutput = 1 if Random < 100000 else 0

  InputImage = Data[Random]
  Conv1 = DisInputToCONV1.ForwardProp(InputImage)
  Conv2 = DisCONV1ToCONV2.ForwardProp(Conv1)
  Input = Conv2.flatten()
  Output = DisInputToOutput.ForwardProp(Input)

  OutputError = RealOutput - Output
  InputError = DisInputToOutput.BackProp(OutputError)
  Conv2Error = InputError.reshape(Conv2.shape)

  Conv1Error = DisCONV1ToCONV2.BackProp(Conv2Error)
  print(Conv1Error.shape)
  DisInputToCONV1.BackProp(Conv1Error)