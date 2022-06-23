#import <math.h>

double *tanh(double *x)
{
  for (int i = 0; i < 4; i++)
  {
    x[i] = tanh(x[i]);
  }
  return x;
}

double *tanhderivative(double *x)
{
  int a;
  for (int i = 0; i < 4; i++)
  {
    a = np.exp(2 * x[i]);
    x[i] = (4 * x[i]) / np.square(x[i] + 1);
  }
  return x;
}