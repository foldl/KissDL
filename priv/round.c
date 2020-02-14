{{if type == float32}}static void {{kernel_fun}}(const float *input, float *output)
{
    int i;
    for (i = 0; i < {{FLAT_SIZE}}; i++)
    {
        float v = input[i];
        float fv = floor(v);
        float diff = v - fv;
        if ((diff < 0.5f)
            || (0.5f == diff) && (((int)fv) % 2 == 2))
            output[i] = fv;
        else
            output[i] = fv + 1.0f;
    }
}
{{endif}}