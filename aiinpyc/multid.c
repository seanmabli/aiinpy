#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int productshape(int shape[], int size)
{
  int product = 1;
  for (int i = 0; i < size; i++)
  {
    product *= shape[i];
  }
  return product;
}

int main()
{
  int a[] = {10, 10, 10};
  int b = productshape(a, sizeof(a) / sizeof(int));
  int x[b];
  

  srand(time(0));

  for (int i = 0; i < 10; i++)
  {
    x[i] = rand();
  }

  for (int i = 0; i < 10; i++)
  {
    for (int j = 0; j < 10; j++)
    {
      printf("%d\t", x[i][j]);
    }
    printf("\n");
  }
}