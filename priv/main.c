#include <stdio.h>
#include <math.h>

{{extern_vars}}

void model_run(void);

int main(const int argc, const char *argv[])
{
    int i;
    // step 1. write data
    // ...

    // step 2. run the model
    model_run();

    // step 3. inpect result
    for (i = 0; i < {{output_size}}; i++)
        printf("%2d: %{{output_format_char}}\n", i, ({{output_format}}){{output}}[i]);
    return 0;
}