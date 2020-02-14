{{if type == float32}}static void {{kernel_fun}}(const float *input, const float *filter, const float *bias, float *output)
{
    int b, in_d, out_y, out_x, f_y, f_x, m;
    for (b = 0; b < {{BATCH_SIZE}}; b++)
    {
        for (out_y = 0; out_y < {{OUTPUT_HEIGHT}}; out_y++)
        {
            const int in_y_origin = (out_y * {{stride_height}}) - {{padding_height}};
            for (out_x = 0; out_x < {{OUTPUT_WIDTH}}; out_x++)
            {
                const int in_x_origin = (out_x * {{stride_width}}) - {{padding_width}};
                for (in_d = 0; in_d < {{INPUT_DEPTH}}; in_d++)
                {
                    for (m = 0; m < {{depth_multiplier}}; m++)
                    {
                        const int out_d = m + in_d * {{depth_multiplier}};
                        float total = 0.0f;
                        for (f_y = 0; f_y < {{FILTER_HEIGHT}}; f_y++)
                        {
                            const int in_y = in_y_origin + {{dilation_height_factor}} * f_y;
                            if ((in_y < 0) || (in_y >= {{INPUT_HEIGHT}})) continue;

                            for (f_x = 0; f_x < {{FILTER_WIDTH}}; f_x++)
                            {
                                const int in_x = in_x_origin + {{dilation_width_factor}} * f_x;
                                if ((in_x < 0) || (in_x >= {{INPUT_WIDTH}})) continue;

                                total += input[in_y * {{INPUT_WIDTH}} * {{INPUT_DEPTH}} + in_x * {{INPUT_DEPTH}} + in_d] *
                                         filter[f_y * {{FILTER_WIDTH}} * {{OUTPUT_DEPTH}} + f_x * {{OUTPUT_DEPTH}} + out_d];
                            }
                        }
                        {{if has_bias}}total += bias[out_d];
                        {{endif}}output[out_y * {{OUTPUT_WIDTH}} * {{OUTPUT_DEPTH}} + out_x * {{OUTPUT_DEPTH}} + out_d] = {{call act total}};
                    }
                }
            }
        }

        // next batch
        input += {{INPUT_HEIGHT}} * {{INPUT_WIDTH}} * {{INPUT_DEPTH}};
        output += {{OUTPUT_HEIGHT}} * {{OUTPUT_WIDTH}} * {{OUTPUT_DEPTH}};
    }
}
{{endif}}