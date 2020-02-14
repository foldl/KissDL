{{if type == float32}}static void {{kernel_fun}}(const float *input, float *output)
{
    int b;
    for (b = 0; b < {{BATCH_SIZE}}; b++)
    {
        float max_coeff = input[0];
        float exp_sum = 0.0;
        float reciprocal_sum;
        int i;
        for (i = 1; i < {{INPUT_SIZE}}; i++)
            if (input[i] > max_coeff) max_coeff = input[i];

        // normalized sum of exps.
        for (i = 0; i < {{INPUT_SIZE}}; i++)
        {
            output[i] = exp((input[i] - max_coeff) * {{beta}});
            exp_sum += output[i];
        }

        // divide by the sum of exps.
        reciprocal_sum = 1.f / exp_sum;
        for (i = 0; i < {{INPUT_SIZE}}; i++)
            output[i] *= reciprocal_sum;

        // next batch
        input += {{INPUT_SIZE}};
        output += {{INPUT_SIZE}};
    }
}
{{endif}}