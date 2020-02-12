// weights is of shape [OUTPUT_SIZE, INPUTSIZE]
{{if type == float32}}void {{kernel_fun}}(const float *input, const float *weights, const float *bias, float *output)
{
    int b;
    for (b = 0; b < {{BATCH_SIZE}}; b++)
    {
        int o;
        for (o = 0; o < {{OUTPUT_SIZE}}; o++)
        {
            int wi;
            float sum = 0;
            const float *ws = weights + o * {{INPUT_SIZE}};
            for (wi = 0; wi < {{INPUT_SIZE}}; wi++)
                sum += input[wi] * ws[wi];
            {{if has_bias}}sum += bias[o];
            {{endif}}output[o] = {{call act sum}};
        }

        input += {{INPUT_SIZE}};
        output += {{OUTPUT_SIZE}};
    }
}
{{endif}}