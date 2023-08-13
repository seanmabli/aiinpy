#include <stdio.h>
#include <stdlib.h>

int main()
{
  int *numbers = malloc(0);
  int size = 0;

  for (int i = 0; i < 3; i++)
  {
    size++;
    numbers = realloc(numbers, size  * sizeof(int));
    numbers[size - 1] = 5;
  }

  for (int ii = 0; ii < 4; ++ii)
  {
    printf("%d\n", numbers[ii]);
  }

  free(numbers);
}