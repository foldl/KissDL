{{if type == float32}}static void {{kernel_fun}}(float *input, float *output)
{
    int b, out_y, out_x, out_d, f_y, f_x;
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
                    const int f_x_start = max(0, in_x_origin);
                    const int f_x_end = min(f_x_start + {{FILTER_WIDTH}}, {{INPUT_WIDTH}});

                    const int f_y_start = max(0, in_y_origin);
                    const int f_y_end = min(f_y_start + {{FILTER_HEIGHT}}, {{INPUT_HEIGHT}});

                    float vmax = input[f_y_start * {{INPUT_WIDTH}} * {{INPUT_DEPTH}} + f_x_start * {{INPUT_DEPTH}} + out_d];
                    for (f_y = f_y_start; f_y < f_y_end; f_y++)
                        for (f_x = f_x_start; f_x < f_x_end; f_x++)
                            vmax = fmax(vmax, input[f_y * {{INPUT_WIDTH}} * {{INPUT_DEPTH}} + f_x * {{INPUT_DEPTH}} + out_d]);

                    output[f_y * {{OUTPUT_WIDTH}} * {{OUTPUT_DEPTH}} + f_x * {{OUTPUT_DEPTH}} + out_d] = {{call act vmax}};
                }
            }
        }

        input += {{INPUT_HEIGHT}} * {{INPUT_WIDTH}} * {{INPUT_DEPTH}};
        output += {{OUTPUT_HEIGHT}} * {{OUTPUT_WIDTH}} * {{OUTPUT_DEPTH}};
    }
}
{{endif}}