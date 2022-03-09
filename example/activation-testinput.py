import src as ai
import numpy as np

prelu = ai.prelu(0.01)

input = np.array([-12673, -36, -1, -0.7, 0, 0.7, 1, 36, 12673])
output = prelu.forward(input)

print(np.round(output))