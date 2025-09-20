import os
import argparse
import tensorflow as tf


def save_tflite(h5_model_path, tflite_save_path, quant_fmt=tf.float16):
    model = tf.keras.models.load_model(h5_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # does float16 quantization for speedup
    converter.target_spec.supported_types = [tf.float16]
    tflite_quant_model = converter.convert()

    # save converted quantization model to tflite format
    with open(tflite_save_path, "wb") as tf_ptr:
        tf_ptr.write(tflite_quant_model)


def main():
    parser = argparse.ArgumentParser(description="Dump hdf5 model to pb file")
    parser.add_argument(
        "-m", "--model_path", type=str, help="path to hdf5/h5 model", default=""
    )
    parser.add_argument(
        "-sp",
        "--save_tflite_dir",
        type=str,
        default="models",
        help="folder path to save tflite file. Default: models",
    )
    parser.add_argument(
        "-p",
        "--save_tflite_name",
        type=str,
        default="model.tflite",
        help="tflite file name. Default: model.tflite",
    )
    args = parser.parse_args()

    tflite_save_path = os.path.join(args.save_tflite_dir, args.save_tflite_name)
    save_tflite(h5_model_path=args.model_path, tflite_save_path=tflite_save_path)


if __name__ == "__main__":
    main()
