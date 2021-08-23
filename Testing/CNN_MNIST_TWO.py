import numpy as np
from emnist import extract_training_samples, extract_test_samples
from TestSrc.NN import NN
from TestSrc.CONV import CONV
from TestSrc.POOL import POOL
from alive_progress import alive_bar
import sys
import wandb

wandb.init(project="CNN_MNIST")

'''
Padding=False & Stride=(1, 1): Done
Padding=True & Stride=(1, 1): Done
Padding=False & Stride=(2, 2): Works Some Of The Time
Padding=True & Stride=(2, 2): Works Some Of The Time
'''

InToConv1 = CONV((4, 3, 3), LearningRate=0.01, Padding=False, Activation='ReLU')
Conv1ToConv2 = CONV((4, 3, 3), LearningRate=0.01, Padding=False, Activation='ReLU')
DenseInToOut = NN(InSize=(4 * 24 * 24), OutSize=10, Activation='StableSoftmax', LearningRate=0.1, WeightsInit=(0, 0))

# Load EMNIST Training An1d Testing Images
TestImageLoaded = 1000
TrainingImages, TrainingLabels = extract_training_samples('digits')
TestImages, TestLabels = extract_test_samples('digits')[0 : TestImageLoaded]

NumOfTrainGen = 5000
with alive_bar(NumOfTrainGen + TestImageLoaded) as bar:
  for Generation in range(NumOfTrainGen):
    # Set Input
    Random = np.random.randint(0, len(TrainingLabels))
    In = (TrainingImages[Random] / 255) - 0.5
    RealOut = np.zeros(10)
    RealOut[TrainingLabels[Random]] = 1
    
    # Forward Propagation
    Conv1 = InToConv1.ForwardProp(In)
    Conv2 = Conv1ToConv2.ForwardProp(Conv1)

    DenseIn = Conv2.flatten()
    Out = DenseInToOut.ForwardProp(DenseIn)

    # Back Propagation
    OutError = RealOut - Out
    DenseInError = DenseInToOut.BackProp(OutError) 
    Conv2Error = DenseInError.reshape(Conv2.shape)

    Conv1Error = Conv1ToConv2.BackProp(Conv2Error)
    InError = InToConv1.BackProp(Conv1Error)
    
    wandb.log({"In": np.sum(abs(In)) / In.size,
               "Conv1": np.sum(abs(Conv1)) / Conv1.size,
               "Conv2": np.sum(abs(Conv2)) / Conv2.size,
               "DenseIn": np.sum(abs(DenseIn)) / DenseIn.size, 
               "Out": np.sum(abs(Out)) / Out.size, 

               "OutError": np.sum(abs(OutError)) / OutError.size, 
               "DenseInError": np.sum(abs(DenseInError)) / DenseInError.size, 
               "Conv2Error": np.sum(abs(Conv2Error)) / Conv2Error.size,
               "Conv1Error": np.sum(abs(Conv1Error)) / Conv1Error.size,
               "InError": np.sum(abs(InError)) / InError.size,
               "Correct": 1 if np.argmax(Out) == TrainingLabels[Random] else 0})

    bar()
  
  NumberCorrect = 0
  for Generation in range(TestImageLoaded):
    In = (TestImages[Generation] / 255) - 0.5
    Conv1 = InToConv1.ForwardProp(In)
    Conv2 = Conv1ToConv2.ForwardProp(Conv1)

    DenseIn = Conv2.flatten()
    Out = DenseInToOut.ForwardProp(DenseIn)
    
    NumberCorrect += 1 if np.argmax(Out) == TestLabels[Generation] else 0

    wandb.log({"Test Correct": 1 if np.argmax(Out) == TestLabels[Generation] else 0})

    bar()

  print(NumberCorrect / TestImageLoaded)