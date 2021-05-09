import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model


# clear prev session
tf.keras.backend.clear_session()


def save_ckpt(model_path, save_ckpt_dir, save_ckpt_name):
    # # clear prev session
    tf.keras.backend.clear_session()
    # Save ckpt file
    model = load_model(model_path)
    session = tf.keras.backend.get_session()

    os.makedirs(save_ckpt_dir, exist_ok=True)
    saver = tf.train.Saver()
    saver.save(session, os.path.join(save_ckpt_dir, save_ckpt_name))


def main(model_path, save_ckpt_dir, save_ckpt_name):
    save_ckpt(model_path, save_ckpt_dir, save_ckpt_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Dump hdf5 model to checkpoint folder')
    parser.add_argument('-m', '--model_path', type=str,
                        help='path to hdf5/h5 model', default="")
    parser.add_argument('-sc', '--save_ckpt_dir', type=str,
                        help='folder path to save ckpt file', default="models")
    parser.add_argument('-c', '--save_ckpt_name', type=str,
                        help='ckpt file name', default="frozen_model.ckpt")
    args = parser.parse_args()

    main(args.model_path,
         args.save_ckpt_dir,
         args.save_ckpt_name)
