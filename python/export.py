import os
import sys
import tensorflow as tf
from keras.layers import Flatten
from keras.models import Model, load_model
from kito import reduce_keras_model  # Ensure kito is installed

# Load the model (output of training - checkpoint)
model = load_model(sys.argv[1])
model_out = sys.argv[2]
save_dir = sys.argv[3]
os.makedirs(save_dir, exist_ok=True)

# Fold batch norms
model_reduced = reduce_keras_model(model)
# Use this model in PC
model_reduced.save(f"{save_dir}/{model_out}.h5")

# Flatten output and save model (Optimize for phone)
output = model_reduced.output
newout = Flatten()(output)
new_model = Model(model_reduced.input, newout)

new_model.save(f"{save_dir}/{model_out}_fin.h5")


# For Float32 Model
converter = tf.lite.TFLiteConverter.from_keras_model_file(
    f"{save_dir}/{model_out}_fin.h5")
tflite_model = converter.convert()
open(f"{save_dir}/{model_out}_fin.tflite", "wb").write(tflite_model)


# For UINT8 Quantization
converter = tf.lite.TFLiteConverter.from_keras_model_file(
    f"{save_dir}/{model_out}_fin.h5")
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.post_training_quantize = True
tflite_model = converter.convert()
open(f"{save_dir}/{model_out}_fin_uint8.tflite", "wb").write(tflite_model)


# For Float16 Quantization (Requires TF 1.15 or above)
try:
    converter = tf.lite.TFLiteConverter.from_keras_model_file(
        f'{save_dir}/{model_out}_fin.h5')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
    tflite_model = converter.convert()
    open(f"{save_dir}/{model_out}_fin_fp16.tflite", "wb").write(tflite_model)
except Exception as e:
    print("Most likely TF version 1.15 or greater is required")
    print(e)


# Sample run: python export.py checkpoints/model.hdf5 reduced_model model_out
