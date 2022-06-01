import tensorflow as tf
from tensorflow import keras
import numpy as np

X = np.array([[0.55, 0.52], [0.57,0.50]])
W = np.array([[[1, 2], [2, 1]]], dtype=np.float64)
X_reshape = X.reshape(1, 2, 2, 1)
weights = [np.asarray([[[[1]], [[2]]], [[[2]], [[1]]]]), np.asarray([0])]

model_Conv2D_Transpose = keras.models.Sequential()
model_Conv2D_Transpose.add(keras.layers.Conv2DTranspose(1, (2, 2), strides=(1, 1), padding='valid', input_shape=(2, 2, 1)))
model_Conv2D_Transpose.set_weights(weights)
optimizer = keras.optimizers.SGD(learning_rate=0.1)

keras_output = model_Conv2D_Transpose.predict(X_reshape).reshape(3, 3)
print(keras_output)

with tf.GradientTape() as tape:
  loss = 1 - keras_output

grads = tape.gradient(loss, model_Conv2D_Transpose.trainable_weights)
optimizer.apply_gradients(zip(grads, model_Conv2D_Transpose.trainable_weights))