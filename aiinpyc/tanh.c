#include <math.h>

double *tanh(double *x, int n)
{
  for (int i = 0; i < n; i++)
  {
    x[i] = tanh(x[i]);
  }
  return x;
}

double *tanhderivative(double *x, int n)
{
  int a;
  for (int i = 0; i < n; i++)
  {
    a = exp(2 * x[i]);
    x[i] = (4 * x[i]) / pow(x[i] + 1, 2);
  }
  return x;
}