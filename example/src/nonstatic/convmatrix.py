import numpy as np

class convmatrix:
  def __init__(self, filtershape, learningrate, padding=False, stride=(1, 1), inshape=None):
    self.learningrate, self.padding, self.stride = learningrate, padding, stride
    self.inshape = inshape
    
    self.filtershape = tuple([1]) + filtershape if len(filtershape) == 2 else filtershape
    
    if inshape is not None:
      self.outshape = tuple([filtershape[0], inshape[0] - filtershape[1] + 1, inshape[1] - filtershape[2] + 1])

      self.filtermatrix = np.zeros((np.prod(inshape), np.prod(self.outshape)))
      self.filter = np.random.uniform(-0.25, 0.25, self.filtershape)

      for f in range(self.outshape[0]):
        for i in range(self.outshape[1]):
          for j in range(self.outshape[2]):
            x = np.zeros(inshape)
            x[i : i + self.filtershape[1], j : j + self.filtershape[2]] = self.filter[f]
            self.filtermatrix[:, f * self.outshape[1] * self.outshape[2] + i * self.outshape[2] + j] = x.flatten()

  def __copy__(self):
    return type(self)(self.filtershape, self.learningrate, self.padding, self.stride, self.inshape)
  
  def __repr__(self):
    return 'convmatrix(inshape=' + str(self.inshape) + ', outshape=' + str(self.outshape) + ', filtershape=' + str(self.filtershape) + ', learningrate=' + str(self.learningrate) + ', padding=' + str(self.padding) + ', stride=' + str(self.stride) + ')'

  def modelinit(self, inshape):
    return self.outshape

  def forward(self, input):
    self.input = input.flatten()
    out = self.input @ self.filtermatrix
    return self.out.reshape(self.outshape)

  def backward(self, outerror):
    outerror = outerror.flatten()
    outgradient = self.derivative * outerror
    # inputerror = np.flip((self.filtermatrix @ outerror).reshape(self.inshape), axis=1)
    self.filtermatrix += np.outer(self.input.T, outgradient) * self.learningrate
    # return inputerror
