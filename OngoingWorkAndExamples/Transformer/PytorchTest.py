import numpy as np
import aiinpy as ai

a = np.array([[1, 2, 3], [4, 5, 6]])
b = a @ np.transpose(a)
c = ai.StableSoftMax(a) @ ai.StableSoftMax(np.transpose(a))
d = ai.StableSoftMax(b)
print(c == d)