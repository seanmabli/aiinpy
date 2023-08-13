#include <stdio.h>
#include "sigmoid.c"
#include "tanh.c"

int main()
{
  double input[4] = {1, 2, 3, 4};
  double *output = tanh(input, sizeof(input) / sizeof(double));

  for (int i = 0; i < 4; i++)
  {
    printf("%f\n", output[i]);
  }
}