import testsrc as ai
import numpy as np

elu = ai.elu(0.01)

input = np.array([-12673, -36, -1, -0.7, 0, 0.7, 1, 36, 12673])
output = elu.forward(input)

print(np.round(output))
