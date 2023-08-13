#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define shape(x) *(x)

int main()
{
  double a[2][2][4] = {{{1, 2, 3, 4}, {5, 6, 7, 8}}, {{1, 2, 3, 4}, {5, 6, 7, 8}}};

  printf("%ld\n", sizeof(a));
  printf("%ld\n", sizeof(*(a)));
  printf("%ld\n", sizeof(*(*(a))));
  printf("%ld\n", sizeof(*(*(*(a)))));
}