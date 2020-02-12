# KissDL

A deep learning model compiler based on K.I.S.S. principle. This tool takes
in a trained model (TFLite model, etc), and generates a cold hard plain C99 implementation.

This tool does two things:

* Memory optimization

    * Using Genetic Algorithm for optimal static memory allocation (see [GAAlloc](https://github.com/foldl/GAAlloc))

* C99 code generation

    * All parameters (except data) are injected into kernels' body

**Note:** This is still under development.

## Build

```erlang
> cd(path of this compiler).
ok
> make:build([all]).
up_to_date
```

## Prepare `emodel`

* Import Tensorflow Lite model

Use this `python` script to convert a TF Lite mode to an `emodel`:

```python
python import_tflite.py tflite_file_name result_file_name
```

## Compile `emodel`:

```erlang
kdl:compile("./examples/tiny_conv_graph.emodel", a_proj_dir)
```

Two C files will be generated under `a_proj_dir`, `main_demo.c` and `model.c`. The whole model is defined in `model.c`.

## Build model

Modify `main_demo.c` to feed in input data, then build the model with simple command:

```bash
gcc main_demo.c model.c
```