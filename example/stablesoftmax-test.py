import src as new
import old
import numpy as np

newSS = new.stablesoftmax()
oldSS = old.stablesoftmax()

print(newSS.forward(new.tensor([1, 2, 3])))
print(oldSS.forward(np.array([1, 2, 3])))

print(newSS.backward(new.tensor([1, 2, 3])))
print(oldSS.backward(np.array([1, 2, 3])))