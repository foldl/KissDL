// about data structure, for each batch:
//      * input is of shape [in_height, in_width, in_depth]
//      * output is of shape [out_height, out_width, out_depth]
// filter is of shape [out_depth, f_height, f_width, in_depth]
{{if type == float32}}_ATTRIBUTE_ void {{kernel_fun}}(const float *input, const float *filter, const float *bias, float *output)
{
    int b, in_d, out_y, out_x, out_d, f_y, f_x, f_d;
    for (b = 0; b < {{BATCH_SIZE}}; b++)
    {
        for (out_y = 0; out_y < {{OUTPUT_HEIGHT}}; out_y++)
        {
            const int in_y_origin = (out_y * {{stride_height}}) - {{padding_height}};
            for (out_x = 0; out_x < {{OUTPUT_WIDTH}}; out_x++)
            {
                const int in_x_origin = (out_x * {{stride_width}}) - {{padding_width}};
                for (out_d = 0; out_d < {{OUTPUT_DEPTH}}; out_d++)
                {
                    float total = 0.f;
                    for (f_y = 0; f_y < {{FILTER_HEIGHT}}; f_y++)
                    {
                        const int in_y = in_y_origin + {{dilation_height_factor}} * f_y;
                        if ((in_y < 0) || (in_y >= {{INPUT_HEIGHT}})) continue;

                        for (f_x = 0; f_x < {{FILTER_WIDTH}}; f_x++)
                        {
                            const int in_x = in_x_origin + {{dilation_width_factor}} * f_x;
                            if ((in_x < 0) || (in_x >= {{INPUT_WIDTH}})) continue;

                            for (in_d = 0; in_d < {{INPUT_DEPTH}}; in_d++)
                            {
                                total += input[in_y * {{INPUT_WIDTH}} * {{INPUT_DEPTH}} + in_x * {{INPUT_DEPTH}} + in_d] *
                                         filter[out_d * {{FILTER_HEIGHT}} * {{FILTER_WIDTH}} * {{FILTER_DEPTH}} + f_y * {{FILTER_WIDTH}} * {{FILTER_DEPTH}} + f_x * {{FILTER_DEPTH}} + in_d];
                            }
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
{{endif}}{{if type in int8 uint8 int16}}_ATTRIBUTE_ void {{kernel_fun}}(const {{c_type}} *input, const {{c_type}} *filter, const int32_t *bias, {{c_type}} *output)
{
    int b, in_d, out_y, out_x, out_d, f_y, f_x, f_d;
    for (b = 0; b < {{BATCH_SIZE}}; b++)
    {
        for (out_y = 0; out_y < {{OUTPUT_HEIGHT}}; out_y++)
        {
            const int in_y_origin = (out_y * {{stride_height}}) - {{padding_height}};
            for (out_x = 0; out_x < {{OUTPUT_WIDTH}}; out_x++)
            {
                const int in_x_origin = (out_x * {{stride_width}}) - {{padding_width}};
                for (out_d = 0; out_d < {{OUTPUT_DEPTH}}; out_d++)
                {
                    int32_t total = 0;
                    for (f_y = 0; f_y < {{FILTER_HEIGHT}}; f_y++)
                    {
                        const int in_y = in_y_origin + {{dilation_height_factor}} * f_y;
                        if ((in_y < 0) || (in_y >= {{INPUT_HEIGHT}})) continue;

                        for (f_x = 0; f_x < {{FILTER_WIDTH}}; f_x++)
                        {
                            const int in_x = in_x_origin + {{dilation_width_factor}} * f_x;
                            if ((in_x < 0) || (in_x >= {{INPUT_WIDTH}})) continue;

                            for (in_d = 0; in_d < {{INPUT_DEPTH}}; in_d++)
                            {
                                const {{c_type}} in = input[in_y * {{INPUT_WIDTH}} * {{INPUT_DEPTH}} + in_x * {{INPUT_DEPTH}} + in_d];
                                const {{c_type}} f = filter[out_d * {{FILTER_HEIGHT}} * {{FILTER_WIDTH}} * {{FILTER_DEPTH}} + f_y * {{FILTER_WIDTH}} * {{FILTER_DEPTH}} + f_x * {{FILTER_DEPTH}} + in_d];
                                total +=  (f + {{filter_offset}}) * (in + {{input_offset}});
                            }
                        }
                    }
                    {{if has_bias}}total += bias[out_d];
                    {{endif}}{{if out_shift >= 0}}total = q31_mult_sat(total << {{out_shift}}, {{output_multiplier}});
                    {{else}}total = q31_mult_sat(total, {{output_multiplier}});
                    total = i32_round_shift_right(total, -{{out_shift}});
                    {{endif}}total = CLIP(total, {{output_activation_min}}, {{output_activation_max}});
                    output[out_y * {{OUTPUT_WIDTH}} * {{OUTPUT_DEPTH}} + out_x * {{OUTPUT_DEPTH}} + out_d] = {{call act total}};
                }
            }
        }
    }

    // next batch
    input += {{INPUT_HEIGHT}} * {{INPUT_WIDTH}} * {{INPUT_DEPTH}};
    output += {{OUTPUT_HEIGHT}} * {{OUTPUT_WIDTH}} * {{OUTPUT_DEPTH}};
}
{{endif}}