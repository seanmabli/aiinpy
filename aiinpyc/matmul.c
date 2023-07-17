#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "array.c"

array matmul(array *a, array *b, array *c)
{
  int *concatshape = malloc(sizeof(a->shape) + sizeof(b->shape));
  int concatshapesize = sizeof(a->shape) + sizeof(b->shape);
  for (int i = 0; i < sizeof(a->shape) / sizeof(int); i++)
  {
    concatshape[i] = a->shape[i];
  }
  for (int i = 0; i < sizeof(b->shape) / sizeof(int); i++)
  {
    concatshape[i + sizeof(a->shape) / sizeof(int)] = b->shape[i];
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
  int similar;
  for (int k = 0; k < uniquesize; k++)
  {
    if (copy[k] > 2)
    {
      copy[k] -= 2;
      similar = unique[k];
    }
  }

  c->shape = unique;
  c->data = malloc(sizeof(*c->data) * unique[0] * unique[1]);

  for (int i = 0; i < unique[0]; i++)
  {
    for (int j = 0; j < unique[1]; j++)
    {
      c->data[i * unique[1] + j] = 0;
      for (int k = 0; k < similar; k++)
      {
        c->data[i * unique[1] + j] += a->data[i * similar + k] * b->data[k * unique[1] + j];
      }
    }
  }

  return *c;
}