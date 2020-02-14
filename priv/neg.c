{{if type in float32 int64 int32 int16 int8}}static void {{kernel_fun}}(const c_type *input, c_type *output)
{
    int i;
    for (i = 0; i < {{FLAT_SIZE}}; i++)
        output[i] = -input[i];
}{{endif}}