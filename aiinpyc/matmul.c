#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

struct array
{
  int *shape;
  double *data;
};

int main()
{
  struct array a = {(int[]){2, 4}, (double[]){1, 2, 3, 4, 5, 6, 7, 8}};
  struct array b = {(int[]){4, 4}, (double[]){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}};

  int *concatshape = malloc(sizeof(a.shape) + sizeof(b.shape));
  int concatshapesize = sizeof(a.shape) + sizeof(b.shape);
  for (int i = 0; i < sizeof(a.shape) / sizeof(int); i++)
  {
    concatshape[i] = a.shape[i];
  }
  for (int i = 0; i < sizeof(b.shape) / sizeof(int); i++)
  {
    concatshape[i + sizeof(a.shape) / sizeof(int)] = b.shape[i];
  }

  int *unique = malloc(0);
  int *copy = malloc(0);
  int uniquesize = 0;

  for (int i = 0; i < concatshapesize / sizeof(int); i++)
  {
    bool found = false;
    int j;
    for (j = 0; j < uniquesize; j++)
    {
      if (concatshape[i] == unique[j])
      {
        found = true;
        break;
      }
    }
    if (!found)
    {
      uniquesize++;
      unique = realloc(unique, uniquesize * sizeof(int));
      copy = realloc(copy, uniquesize * sizeof(int));
      unique[uniquesize - 1] = concatshape[i];
      copy[uniquesize - 1] = 1;
    }
    else
    {
      copy[j]++;
    }
  }

  for (int k = 0; k < uniquesize; k++)
  {
    if (copy[k] > 2)
    {
      copy[k] -= 2;
    }
  }

  int c[unique[0], unique[1]];

  for (int i = 0; i < unique[0]; i++)
  {
    for (int j = 0; j < unique[1]; j++)
    {
      c[i, j] = a[i, j] + b[i, j];
    }
  }

  free(concatshape);
  free(unique);
  free(copy);
}