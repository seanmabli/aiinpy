import numpy as np
import mnist
import aiinpy as ai

InputImageToConv1 = ai.CONV((4, 3, 3), LearningRate=0.005)
Conv1ToPool1 = ai.POOL(2)
InputToHid1 = ai.NN((4 * 13 * 13), 10, "StableSoftMax", LearningRate=0.1, WeightsInit=(0, 0))

def TestNetwork():
  NumberCorrect = 0
  Generation = 0
  for i in range(TestImageLoaded):
    InputImage = (TestImages[i] / 255) - 0.5

    ConvolutionLayer1 = InputImageToConv1.ForwardProp(InputImage)
    MaxPooling1 = Conv1ToPool1.ForwardProp(ConvolutionLayer1)
    Input = MaxPooling1.flatten()
    Output = InputToHid1.ForwardProp(Input)
    
    NumberCorrect += 1 if np.argmax(Output) == TestLabels[i] else 0
    Generation += 1
  return NumberCorrect / Generation

# Load MNIST Training And Testing Images                               
TrainingImageLoaded = 1000
TestImageLoaded = 1000
TrainingImages = mnist.train_images()[0:TrainingImageLoaded]
TrainingLabels = mnist.train_labels()[0:TrainingImageLoaded]
TestImages = mnist.test_images()[0:TestImageLoaded]
TestLabels = mnist.test_labels()[0:TestImageLoaded]

for Generation in range(1000):
  # Set Input
  Random = np.random.randint(0, TrainingImageLoaded)
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
  
print(TestNetwork())