import os
import argparse
import tensorflow as tf


def save_pb(model_path, save_pb_dir):
    model = tf.keras.models.load_model(model_path)
    input_name = model.input
    output_name = model.output
    print(f"input_name: {input_name}")
    print(f"output_name: {output_name}")

    os.makedirs(save_pb_dir, exist_ok=True)
    model.save(save_pb_dir)


def main(model_path, save_pb_dir):
    save_pb(model_path, save_pb_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump hdf5 model to pb file")
    parser.add_argument(
        "-m", "--model_path", type=str, help="path to hdf5/h5 model", default=""
    )
    parser.add_argument(
        "-sp",
        "--save_pb_dir",
        type=str,
        help="folder path to save pb file",
        default="models/savedmodel",
    )
    args = parser.parse_args()

    main(args.model_path, args.save_pb_dir)
