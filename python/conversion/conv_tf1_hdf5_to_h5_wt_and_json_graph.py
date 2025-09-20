import argparse
from tensorflow.keras.models import load_model


def convert(model_path, weight_save_path, json_save_path):
    model = load_model(model_path)
    model.save_weights(weight_save_path)
    json_config = model.to_json()

    with open(json_save_path, "w", encoding="utf-8") as json_file:
        json_file.write(json_config)

    print(f"Saved weights to {weight_save_path} and json config to {json_save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="hdf5/h5 model to json config and weight only h5 file"
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        help="path to hdf5/h5 model",
        default="models/transpose_seg/deconv_bnoptimized_munet.h5",
    )
    parser.add_argument(
        "-w",
        "--weight_save_path",
        type=str,
        help="folder path to save model weights",
        default="models/transpose_seg/deconv_bnoptimized_munet_weights.h5",
    )
    parser.add_argument(
        "-j",
        "--json_save_path",
        type=str,
        help="folder path to save json config",
        default="models/transpose_seg/deconv_bnoptimized_munet_config.json",
    )

    args = parser.parse_args()
    convert(args.model_path, args.weight_save_path, args.json_save_path)


if __name__ == "__main__":
    main()
