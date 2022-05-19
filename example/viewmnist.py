from emnist import extract_training_samples, extract_test_samples
from matplotlib import pyplot

trainx, trainy = extract_training_samples('digits')

for i in range(25):
	pyplot.subplot(5, 5, 1 + i)
	pyplot.axis('off')
	pyplot.imshow(trainx[i], cmap='gray_r')

pyplot.show()