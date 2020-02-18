## Summay of Examples

1. tiny_conv_graph.tflite

    A trained Tensorflow [speech_commands](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands) model.

    Trained with following parameters:

    ```
    python train.py --model_architecture=tiny_conv --window_stride=20 --preprocess=micro --wanted_words="on,off" --silence_percentage=25 --unknown_percentage=25 --verbosity=INFO --how_many_training_steps="15000,3000" --learning_rate="0.001,0.0001"
    ```

    Freezed by:

    ```
    python freeze.py --start_checkpoint=\tmp\speech_commands_train\tiny_conv.ckpt-18000 --output_file=\tmp\tiny_conv_graph.pb --model_architecture=tiny_conv --preprocess=micro --window_stride=20
    ```

1. tiny_embedding_conv_graph.tflite

    A trained Tensorflow [speech_commands](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands) model.

    Trained with following parameters:

    ```
    python train.py --model_architecture=tiny_embedding_conv --window_stride=20 --preprocess=micro --wanted_words="on,off" --silence_percentage=25 --unknown_percentage=25 --verbosity=INFO --how_many_training_steps="15000,3000" --learning_rate="0.001,0.0001"
    ```

    Freezed by:

    ```
    python freeze.py --start_checkpoint=\tmp\speech_commands_train\tiny_embedding_conv.ckpt-18000 --output_file=\tmp\tiny_embedding_conv_graph.pb --model_architecture=tiny_embedding_conv --preprocess=micro --window_stride=20
    ```

1. conv_graph.tflite

    A trained Tensorflow [speech_commands](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands) model like above, but using the "conv" model, and "MFCC" feature.