#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void)
{
    const char *string_table[] = { // array of pointers to constant strings
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon"
    };
    int table_size = 5; // This must match the number of entries above

    srand(time(NULL)); // randomize the start value

    for (int i = 1; i <= 10; ++i)
    {
        const char *rand_string = string_table[rand() % table_size];
        printf("%2d. %s\n", i,  rand_string);
    }

    return 0;
}