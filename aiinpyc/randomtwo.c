#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int *random()
{

  static int r[10];

  srand(time(0));

  for (int i = 0; i < 10; ++i)
  {
    r[i] = rand();
  }

  return r;
}

int main()
{
  int i;
  int *p = random();

  for (int i = 0; i < 10; i++)
  {
    printf("%d\n", p[i]);
  }

  return 0;
}