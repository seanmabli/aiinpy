#include <math.h>

double *sigmoid(double *x)
{
  for (int i = 0; i < 4; i++)
  {
    x[i] = 1 / (1 + exp(-x[i]));
  }
  return x;
}

double *sigmoidderivative(double *x)
{
  for (int i = 0; i < 4; i++)
  {
    x[i] = exp(-x[i]) / pow(1 + exp(-x[i]), 2);
  }
  return x;
}