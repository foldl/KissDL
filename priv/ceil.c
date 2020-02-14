{{if type == float32}}static void {{kernel_fun}}(const float *input, float *output)
{
    int i;
    for (i = 0; i < {{FLAT_SIZE}}; i++)
        output[i] = ceil(input[i]);
}
{{endif}}