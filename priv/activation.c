{{if type == float32}}{{if activation == none}}{{value}}{{endif}}{{if
    activation == relu}}{{value}} > 0.0f ? {{value}} : 0.0{{endif}}{{if
    activation == relu1}}CLIP({{value}}, -1.0f, 1.0f){{endif}}{{if
    activation == relu6}}CLIP({{value}}, 0.0f, 6.0f){{endif}}{{if
    activation == tanh}}tanh({{value}}){{endif}}{{if
    activation == sign}}signbit({{value}}) ? 1.0f : 0.0f{{endif}}{{if
    activation == sigmoid}}1.0f / (1.0f + exp(-{{value}})){{endif}}{{endif}}{{if
    type in int32 int16 int8}}{{if activation == none}}{{value}}{{endif}}{{if
    activation == relu}}{{value}} > 0 ? {{value}} : 0{{endif}}{{if
    activation == relu1}}CLIP({{value}}, -1, 1){{endif}}{{if
    activation == relu6}}CLIP({{value}}, 0, 6){{endif}}{{endif}}