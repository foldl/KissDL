{{if type == float32}}static void {{kernel_fun}}(const c_type *input, c_type *output)
{
    int i;
    for (i = 0; i < {{FLAT_SIZE}}; i++)
        output[i] = floor(input[i]);
}
{{endif}}