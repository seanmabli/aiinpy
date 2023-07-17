#include <stdio.h>
#include "matmul.c"

typedef struct
{
  unsigned int inshape;
  unsigned int outshape;
  array *weights;
  array *biases;
  double learningrate;
  char *activation[];
} nn;

void init(nn *model, unsigned int numoflayers) {
  
}

array forward(array *in, nn *model, unsigned int numoflayers)
{
  for (int i = 0; i < numoflayers; i++)
  {
    matmul(in, model[i].weights, in);
  }
}

/*
self.input = input.flatten()
out = self.weights.T @ self.input + self.biases
self.out = self.activation.forward(out)
self.derivative = self.activation.backward(out) # now it applys the derivative to the output without the activation function, check if this is right
return self.out.reshape(self.outshape)
*/