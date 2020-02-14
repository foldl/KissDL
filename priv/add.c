{{if broadcast}}{{if type in float32}}static void {{kernel_fun}}(const c_type *input1, const c_type *input2, c_type *output)
{
}{{endif}}
{{else}}{{if type == float32}}static void {{kernel_fun}}(const float *input1, const float *input2, float *output)
{
    int i = 0;
    for (; i < {{FLAT_SIZE}}; i++)
    {
        float total = input1[i] + input2[i];
        output[i] = {{call act total}};
    }
}{{endif}}
{{endif}}