#include <stdio.h>
#include <time.h>
#include "nn.c"

int main()
{
  // random init
  srand(time(NULL));

  // define model
  nn layer1;
  layer1.inshape = 2;
  layer1.outshape = 16;
  layer1.learningrate = 0.01;
  layer1.activation[7] = "sigmoid";

  nn layer2;
  layer2.inshape = 16;
  layer2.outshape = 16;
  layer2.learningrate = 0.01;
  layer2.activation[7] = "sigmoid";

  nn layer3;
  layer3.inshape = 16;
  layer3.outshape = 2;
  layer3.learningrate = 0.01;
  layer3.activation[7] = "sigmoid";

  nn model[3] = {layer1, layer2, layer3};

  // train
  for (int gen = 0; gen < 12000; gen++)
  {
    // generate input
    array in = {(int[]){2}, (double[]){rand() & 1, rand() & 1}};
    if (arraysum(&in) == 1)
    {
      array out = {(int[]){2}, (double[]){1, 0}};
    }
    else
    {
      array out = {(int[]){2}, (double[]){0, 1}};
    }

    // forward

  }

  /*
  // matmul example:
  array a = {(int[]){2, 4}, (double[]){1, 2, 3, 4, 5, 6, 7, 8}};
  array b = {(int[]){4, 4}, (double[]){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}};
  array c;
  matmul(&a, &b, &c);

  for (int i = 0; i < c.shape[0]; i++)
  {
    for (int j = 0; j < c.shape[1]; j++)
    {
      printf("%f ", c.data[i * c.shape[1] + j]);
    }
    printf("\n");
  }
  */
}