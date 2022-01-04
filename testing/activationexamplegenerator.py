import testsrc as ai

for activation in ['binarystep', 'gaussian', 'identity', 'leakyrelu', 'mish', 'relu', 'selu', 'sigmoid', 'silu', 'softmax', 'softplus', 'stablesoftmax', 'tanh']:
  print(eval('ai.' + activation + '()'))