# KissDL

A deep learning model compiler based on K.I.S.S. principle. This tool takes
in a trained model, and generates a cold hard plain C99 implementation.

Comparing to Tensorflow Lite, this compiler does two things that are missing in TF Lite:

* Memory optimization

    * Using Genetic Algorithm for optimal static memory allocation (see [GAAlloc](https://github.com/foldl/GAAlloc))

* C99 code generation

    * Parameters are pre-computed and injected into kernels' body

The trained model is saved as an _emodel_, which is just a collection of readable [Erlang](https://erlang.org)
expressions, plus raw data files. _emodel_ can be generated from TF Lite model.

**Note:** This is still under development.

## Build

```erlang
> cd(path of this compiler).
ok
> make:all([all]).
...
up_to_date
```

## Prepare _emodel_

* Import Tensorflow Lite model

Use this `python` script to convert a TF Lite mode to an _emodel_. Before using this script,
make sure you already have the `tflite` python library, which can be generated by the
[FlatBuffers](https://google.github.io/flatbuffers/) compiler.

```python
python import_tflite.py ./examples/tiny_conv_graph.tflite ./examples/tiny_conv_graph.emodel
```

## Compile _emodel_:

```erlang
kdl:compile("./examples/tiny_conv_graph.emodel", a_proj_dir)
```

Two C files will be generated under `a_proj_dir`, `main_demo.c` and `model.c`. The whole model is defined in `model.c`.

## Build model

Modify `main_demo.c` to feed in input data, then build the model with simple command:

```bash
gcc main_demo.c model.c
```