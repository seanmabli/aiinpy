#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double *sigmoid(double *x)
{
  for (int i = 0; i < 4; i++)
  {
    x[i] = 1 / (1 + exp(-x[i]));
  }
  return x;
}

int main()
{
  double input[4] = {1, 2, 3, 4};
  double *output = sigmoid(input);

  for (int i = 0; i < 4; i++)
  {
    printf("%f\n", output[i]);
  }
}