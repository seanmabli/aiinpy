
# example of training a gan on mnist
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot

def generate_real_samples(dataset, n_samples):
	ix = randint(0, dataset.shape[0], n_samples)
	X = dataset[ix]
	y = ones((n_samples, 1))
	return X, y
 
def generate_latent_points(latent_dim, n_samples):
	x_input = randn(latent_dim * n_samples)
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
 
def generate_fake_samples(g_model, latent_dim, n_samples):
	x_input = generate_latent_points(latent_dim, n_samples)
	X = g_model.predict(x_input)
	y = zeros((n_samples, 1))
	return X, y

def save_plot(examples, epoch, n=10):
	for i in range(n * n):
		pyplot.subplot(n, n, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()

# size of the latent space
latent_dim = 100

# create the discriminator
d_model = Sequential()
d_model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=(28, 28, 1)))
d_model.add(LeakyReLU(alpha=0.2))
d_model.add(Dropout(0.4))
d_model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
d_model.add(LeakyReLU(alpha=0.2))
d_model.add(Dropout(0.4))
d_model.add(Flatten())
d_model.add(Dense(1, activation='sigmoid'))
opt = Adam(lr=0.0002, beta_1=0.5)
d_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# create the generator
g_model = Sequential()
n_nodes = 128 * 7 * 7
g_model.add(Dense(n_nodes, input_dim=latent_dim))
g_model.add(LeakyReLU(alpha=0.2))
g_model.add(Reshape((7, 7, 128)))
g_model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
g_model.add(LeakyReLU(alpha=0.2))
g_model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
g_model.add(LeakyReLU(alpha=0.2))
g_model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))

# create the gan
d_model.trainable = False
gan_model = Sequential()
gan_model.add(g_model)
gan_model.add(d_model)
opt = Adam(lr=0.0002, beta_1=0.5)
gan_model.compile(loss='binary_crossentropy', optimizer=opt)

# load image data
(trainX, _), (_, _) = load_data()
dataset = expand_dims(trainX, axis=-1)
dataset = dataset.astype('float32')
dataset = dataset / 255.0

# train model
n_epochs = 100
n_batch = 256
n_samples = 100
bat_per_epo = int(dataset.shape[0] / n_batch)
half_batch = int(n_batch / 2)

for i in range(n_epochs):
	for j in range(bat_per_epo):
		X_real, y_real = generate_real_samples(dataset, half_batch)
		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
		d_loss, _ = d_model.train_on_batch(X, y)
		X_gan = generate_latent_points(latent_dim, n_batch)
		y_gan = ones((n_batch, 1))
		g_loss = gan_model.train_on_batch(X_gan, y_gan)
		print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
	if (i+1) % 10 == 0:
		X_real, y_real = generate_real_samples(dataset, n_samples)
		_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
		x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
		_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
		print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
		save_plot(x_fake, i)
		filename = 'generator_model_%03d.h5' % (i + 1)
		g_model.save(filename)