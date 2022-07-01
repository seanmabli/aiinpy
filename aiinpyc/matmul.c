#include <stdio.h>
#include <stdlib.h>
#include <math.h>

struct array
{
  int *shape;
  double *data;
};

int main()
{
  struct array a = {(int[]){2, 4}, (double[][]){{1, 2, 3, 4}, {5, 6, 7, 8}}};
}