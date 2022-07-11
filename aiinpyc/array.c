typedef struct
{
  unsigned int *shape;
  double *data;
} array;

array arrayrandom(array *a)
{
  srand(time(NULL));
  
  for (int i = 0; i < a->shape[0] * a->shape[1]; i++)
  {
    a->data[i] = (double)rand() / (double)RAND_MAX;
  }
}

int arraysum(array *a)
{
  int sum = 0;
  for (int i = 0; i < a->shape[0]; i++)
  {
    sum += a->data[i];
  }
  return sum;
}