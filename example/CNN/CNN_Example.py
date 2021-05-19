import numpy as np
from emnist import extract_training_samples, extract_test_samples
from NN import NN
from CONV import CONV
import aiinpy as ai
from alive_progress import alive_bar

InputImageToConv1 = CONV((4, 3, 3), LearningRate=0.005, Padding='Same')
Conv1ToPool1 = ai.POOL(2)
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
    ConvolutionLayer1 = InputImageToConv1.ForwardProp(InputImage)
    MaxPooling1 = Conv1ToPool1.ForwardProp(ConvolutionLayer1)
    Input = MaxPooling1.flatten()
    Output = InputToHid1.ForwardProp(Input)

    # Back Propagation
    OutputError = RealOutput - Output
    InputError = InputToHid1.BackProp(OutputError) 
    MaxPooling1Error = InputError.reshape(MaxPooling1.shape)
    
    ConvolutionalError = Conv1ToPool1.BackProp(MaxPooling1Error)
    InputImageToConv1.BackProp(ConvolutionalError)
    
    bar()
  
  NumberCorrect = 0
  for Generation in range(TestImageLoaded):
    InputImage = (TestImages[Generation] / 255) - 0.5
    ConvolutionLayer1 = InputImageToConv1.ForwardProp(InputImage)
    MaxPooling1 = Conv1ToPool1.ForwardProp(ConvolutionLayer1)
    Input = MaxPooling1.flatten()
    Output = InputToHid1.ForwardProp(Input)
    
    NumberCorrect += 1 if np.argmax(Output) == TestLabels[Generation] else 0
    bar()

  print(NumberCorrect / Generation)