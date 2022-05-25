from tensorflow import keras
import numpy as np
from math import floor, ceil
from src import convtranspose, identity
import time

def Conv2DTranspose(X, W, padding="valid", strides=(1, 1)):
    # Define output shape before padding
    row_num = (X.shape[0] - 1) * strides[0] + W.shape[0]
    col_num = (X.shape[1] - 1) * strides[1] + W.shape[1]
    output = np.zeros([row_num, col_num])
    # Calculate the output
    for i in range(0, X.shape[0]):
        i_prime = i * strides[0] # Index in output
        for j in range(0, X.shape[1]):
            j_prime = j * strides[1]
            # Insert values
            for k_row in range(W.shape[0]):
                for k_col in range(W.shape[1]):
                    output[i_prime+k_row, j_prime+k_col] += W[k_row, k_col] * X[i, j]
    # Define length of padding
    if padding == "same":
        # returns the output with the shape of (input shape)*(stride)
        p_left = floor((W.shape[0] - strides[0])/2)
        p_right = W.shape[0] - strides[0] - p_left
        p_top = floor((W.shape[1] - strides[1])/2)
        p_bottom = W.shape[1] - strides[1] - p_left
    elif padding == "valid":
        # returns the output without any padding
        p_left = 0
        p_right = 0
        p_top = 0
        p_bottom = 0
    # Add padding
    print(output.shape)
    output_padded = output[p_left:output.shape[0]-p_right, p_top:output.shape[0]-p_bottom]
    print(output_padded.shape)
    return(np.array(output_padded))

timea = time.time()
X = np.array([[55, 52], [57,50]])
X_reshape = X.reshape(1, 2, 2, 1)
W = np.array([[1, 2], [2, 1]])
my_output = Conv2DTranspose(X, W, padding="valid", strides=(1, 1))
open('a.txt', 'w').write("Kuan Wei Conv2DTranspose:" + str(time.time() - timea) + " \n" + str(my_output) + "\n\n")

timeb = time.time()
X = np.array([[55, 52], [57,50]])
W = np.array([[1, 2], [2, 1]])
modela = convtranspose((2, 2), (2, 2), padding=False, stride=(1, 1), learningrate=0, activation=identity())
modela.Filter = W
open('a.txt', 'a').write("aiinpy Conv2DTranspose:" + str(time.time() - timeb) + " \n" + str(modela.forward(X)) + "\n\n")

model_Conv2D_Transpose = keras.models.Sequential()
model_Conv2D_Transpose.add(keras.layers.Conv2DTranspose(1, (2, 2), strides=(1, 1), padding='valid', input_shape=(2, 2, 1)))
weights = [np.asarray([[[[1]], [[2]]], [[[2]], [[1]]]]), np.asarray([0])]
model_Conv2D_Transpose.set_weights(weights)
timec = time.time()
keras_output = model_Conv2D_Transpose.predict(X_reshape)
keras_output = keras_output.reshape(3, 3)
open('a.txt', 'a').write("Keras Conv2DTranspose:" + str(time.time() - timec) + " \n" + str(keras_output) + "\n\n") 

X = np.array([[55, 52], [57,50]])
X_reshape = X.reshape(1, 2, 2, 1)
W = np.array([[1, 2], [2, 1]])
my_output = Conv2DTranspose(X, W, padding="same", strides=(1, 1))
open('a.txt', 'a').write("Kuan Wei Conv2DTranspose: \n" + str(my_output) + "\n\n")

X = np.array([[55, 52], [57,50]])
W = np.array([[1, 2], [2, 1]])
modelb = convtranspose((2, 2), (2, 2), padding=True, stride=(1, 1), learningrate=0, activation=identity())
modelb.Filter = W
open('a.txt', 'a').write("aiinpy Conv2DTranspose: \n" + str(modelb.forward(X)) + "\n\n")

model_Conv2D_Transpose = keras.models.Sequential()
model_Conv2D_Transpose.add(keras.layers.Conv2DTranspose(1, (2, 2), strides=(1, 1), padding='same', input_shape=(2, 2, 1)))
weights = [np.asarray([[[[1]], [[2]]], [[[2]], [[1]]]]), np.asarray([0])]
model_Conv2D_Transpose.set_weights(weights)
keras_output = model_Conv2D_Transpose.predict(X_reshape)
keras_output = keras_output.reshape(2, 2)
open('a.txt', 'a').write("Keras Conv2DTranspose: \n" + str(keras_output))