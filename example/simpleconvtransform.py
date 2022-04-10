import numpy as np

class convtranspose:
  def __init__(self, inshape, filtershape):
    self.inshape, self.filtershape = inshape, filtershape
    if len(inshape) == 2:
      inshape = tuple([self.filtershape[0]]) + inshape
    self.outshape = np.array([filtershape[0], , dtype=np.int)
    self.out = np.zeros(self.outshape)

    self.Filter = np.random.uniform(-0.25, 0.25, (self.filtershape))
    self.bias = np.zeros(self.filtershape[0])

  def forward(self, input):
    self.input = input
    if(input.ndim == 2):
      self.input = np.stack(([self.input] * self.filtershape[0]))

    self.out = np.zeros(self.outshape)
    for i in range(0, self.inshape[1]):
      for j in range(0, self.inshape[2]):
        self.out[:, i * self.stride[0] : i * self.stride[0] + self.filtershape[1], j * self.stride[1] : j * self.stride[1] + self.filtershape[2]] += self.input[:, i, j][:, np.newaxis, np.newaxis] * self.Filter

    self.out += self.bias[:, np.newaxis, np.newaxis]
    self.out = self.activation.forward(self.out)

    if self.padding == True:
      self.out = self.out[:, 1 : self.outshape[1] - 1, 1 : self.outshape[2] - 1]

    return self.out

model = convtranspose((3, 3), (3, 3))
input = np.random.uniform(-0.25, 0.25, (3, 3))
output = model.forward(input)
print(output)