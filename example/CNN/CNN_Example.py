import numpy as np
from emnist import extract_training_samples, extract_test_samples
from NN import NN
from CONV import CONV
from POOL import POOL
from alive_progress import alive_bar
import sys
import time

InputImageToConv1 = CONV((4, 3, 3), LearningRate=0.01, Padding=True, Activation='ReLU')
Conv1ToConv2 = CONV((4, 3, 3), LearningRate=0.01, Padding=True, Activation='ReLU')
Conv2ToMax1 = POOL(2)
InputToHid1 = NN(InputSize=(4 * 14 * 14), OutputSize=10, Activation='StableSoftMax', LearningRate=0.1, WeightsInit=(0, 0))

# Load EMNIST Training And Testing Images
TestImageLoaded = 1000
TrainingImages, TrainingLabels = extract_training_samples('digits')
TestImages, TestLabels = extract_test_samples('digits')[0 : TestImageLoaded]

NumOfTrainGen = 5000
with alive_bar(NumOfTrainGen + TestImageLoaded) as bar:
  for Generation in range(NumOfTrainGen):
    # Set Input
    Random = np.random.randint(0, len(TrainingLabels))
    InputImage = (TrainingImages[Random] / 255) - 0.5
    RealOutput = np.zeros(10)
    RealOutput[TrainingLabels[Random]] = 1
    
    # Forward Propagation
    Conv1 = InputImageToConv1.ForwardProp(InputImage)
    Conv2 = Conv1ToConv2.ForwardProp(Conv1)
    Max1 = Conv2ToMax1.ForwardProp(Conv2)

    Input = Max1.flatten()
    Output = InputToHid1.ForwardProp(Input)

    # Back Propagation
    OutputError = RealOutput - Output
    InputError = InputToHid1.BackProp(OutputError) 
    Max1Error = InputError.reshape(Max1.shape)

    Conv2Error = Conv2ToMax1.BackProp(Max1Error)
    Conv1Error = Conv1ToConv2.BackProp(Conv2Error)
    InputImageError = InputImageToConv1.BackProp(Conv1Error)

    bar()
  
  InputToHid1.ChangeDropoutRate(0)
  NumberCorrect = 0
  for Generation in range(TestImageLoaded):
    InputImage = (TestImages[Generation] / 255) - 0.5
    Conv1 = InputImageToConv1.ForwardProp(InputImage)
    Conv2 = Conv1ToConv2.ForwardProp(Conv1)
    Max1 = Conv2ToMax1.ForwardProp(Conv2)
    Input = Max1.flatten()
    Output = InputToHid1.ForwardProp(Input)
    
    NumberCorrect += 1 if np.argmax(Output) == TestLabels[Generation] else 0
    bar()

  print(NumberCorrect / TestImageLoaded)