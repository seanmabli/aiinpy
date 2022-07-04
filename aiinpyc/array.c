typedef struct
{
  unsigned int *shape;
  double *data;
} array;

int arraysum(array *a)
{
  int sum = 0;
  printf("%ld\n", sizeof(a->shape));
  for (int i = 0; i < a->shape[0]; i++)
  {
    sum += a->data[i];
  }
  return sum;
}