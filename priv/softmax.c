{{if type == float32}}void {{kernel_fun}}(const float* in, float* out)
{
    int b;
    for (b = 0; b < {{BATCH_SIZE}}; b++)
    {
        float max_coeff = in[0];
        float exp_sum = 0.0;
        float reciprocal_sum;
        int i;
        for (i = 1; i < {{INPUT_SIZE}}; i++)
            if (in[i] > max_coeff) max_coeff = in[i];

        // normalized sum of exps.
        for (i = 0; i < {{INPUT_SIZE}}; i++)
        {
            out[i] = exp((in[i] - max_coeff) * {{beta}});
            exp_sum += out[i];
        }

        // divide by the sum of exps.
        reciprocal_sum = 1.f / exp_sum;
        for (i = 0; i < {{INPUT_SIZE}}; i++)
            out[i] *= reciprocal_sum;

        // next batch
        in += {{INPUT_SIZE}};
        out += {{INPUT_SIZE}};
    }
}
{{endif}}