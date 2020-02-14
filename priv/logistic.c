{{if type == float32}}static void {{kernel_fun}}(const float* in, float* out)
{
    int i;
    for (i = 0; i < {{FLAT_SIZE}}; i++)
    {
        float v = input[i];
        if (v > 16.619047164916992188f)
            v = 1.0f;
        else if (v < -9.0f)
            v = exp(v);
        else
            v = 1.0f / (1.0f + exp(-v));
        out[i] = v;
    }
}
{{endif}}