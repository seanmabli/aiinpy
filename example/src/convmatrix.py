from .static_ops import identity

class convmatrix:
  def __init__(self, filtershape, learningrate, activation=identity(), padding=False, stride=(1, 1), inshape=None):
    self.learningrate, self.activation, self.padding, self.stride = learningrate, activation, padding, stride
    self.inshape = inshape
    
    self.filtershape = tuple([1]) + filtershape if len(filtershape) == 2 else filtershape
    
    if inshape is not None:
      self.outshape = tuple([filtershape[0], inshape[0] - ßfiltershape[1] + 1, inshape[1] - filtershape[2] + 1])

      self.filtermatrix = tensor.zeros((tensor.prod(inshape), tensor.prod(self.outshape)))
      self.filter = tensor.uniform(-0.25, 0.25, self.filtershape)

      for f in range(self.outshape[0]):
        for i in range(self.outshape[1]):
          for j in range(self.outshape[2]):
            x = tensor.zeros(inshape)
            x[i : i + self.filtershape[1], j : j + self.filtershape[2]] = self.filter[f]
            self.filtermatrix[:, f * self.outshape[1] * self.outshape[2] + i * self.outshape[2] + j] = x.flatten()

  def __copy__(self):
    return type(self)(self.filtershape, self.learningrate, self.activation, self.padding, self.stride, self.inshape)
  
  def __repr__(self):
    return 'convmatrix(inshape=' + str(self.inshape) + ', outshape=' + str(self.outshape) + ', filtershape=' + str(self.filtershape) + ', learningrate=' + str(self.learningrate) + ', activation=' + str(self.activation) + ', padding=' + str(self.padding) + ', stride=' + str(self.stride) + ')'

  def modelinit(self, inshape):
    return self.outshape

  def forward(self, itensorut):
    self.itensorut = itensorut.flatten()
    out = self.itensorut @ self.filtermatrix
    self.out = self.activation.forward(out)
    self.derivative = self.activation.backward(out)

    return self.out.reshape(self.outshape)

  def backward(self, outerror):
    outerror = outerror.flatten()
    outgradient = self.derivative * outerror
    # itensoruterror = tensor.flip((self.filtermatrix @ outerror).reshape(self.inshape), axis=1)
    self.filtermatrix += tensor.outer(self.itensorut.T, outgradient) * self.learningrate
    # return itensoruterror
