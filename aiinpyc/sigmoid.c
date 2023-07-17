#include <math.h>

double *sigmoid(double *x, int n)
{
  for (int i = 0; i < n; i++)
  {
    x[i] = 1 / (1 + exp(-x[i]));
  }
  return x;
}

double *sigmoidderivative(double *x, int n)
{
  for (int i = 0; i < n; i++)
  {
    x[i] = exp(-x[i]) / pow(1 + exp(-x[i]), 2);
  }
  return x;
}