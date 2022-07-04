#include <stdio.h>
#include "matmul.c"

typedef struct
{
  unsigned int inshape;
  unsigned int outshape;
  double *weights;
  double *biases;
  double learningrate;
  char *activation[];
} nn;

array forward(array *in, nn *model) {
  array out;
  out.shape[0] = model->outshape;
  out.shape[1] = 1;
  out.data = malloc(sizeof(double) * out.shape[0]);
  for (int i = 0; i < out.shape[0]; i++) {
    out.data[i] = 0;
  }
  for (int i = 0; i < model->inshape; i++) {
    for (int j = 0; j < model->outshape; j++) {
      out.data[j] += in->data[i] * model->weights[i * model->outshape + j];
    }
  }
  for (int i = 0; i < model->outshape; i++) {
    out.data[i] += model->biases[i];
  }
  for (int i = 0; i < model->outshape; i++) {
    if (strcmp(model->activation[i], "sigmoid") == 0) {
      out.data[i] = sigmoid(out.data[i]);
    }
  }
  return out;
}