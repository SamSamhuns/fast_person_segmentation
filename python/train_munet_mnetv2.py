from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from utils.preprocess import calc_mean_std, normalize_batch
from modules.mobileunet_mobilenet_v2.munet_mnetv2 import get_mobile_unet_mnetv2

# TRAIN CONFIGURATION SETTINGS #######################################################
# set PRETRAINED_MODEL_PATH to None to train from scratch
MODEL_TYPE = "transpose"  # bilinear or transpose for the Decoder

PRETRAINED_MODEL_PATH = None  # 'checkpoints/deconv_bnoptimized_munet.h5'
FINETUNE = False

OPTIMIZER = Adam
OPTIM_PARAMS = {"learning_rate": 1e-3}  # passed as dict to TRAIN_OPTIMIZER
EPOCHS = 1
BATCH_SIZE = 32
TRAIN_FRACTION = 0.8
VAL_FRACTION = 1 - TRAIN_FRACTION
Q_AWARE_TRAIN = True  # quantization aware train mode

CHECKPOINT_FMT = "checkpoints/deconv_bnoptimized_munet-{epoch:02d}-{val_loss:.2f}.hdf5"
LOG_PATH = "./logs"
# ####################################################################################


if MODEL_TYPE not in {"transpose", "bilinear"}:
    raise AttributeError("MODEL_TYPE must be transpose or bilinear")

# Load the dataset
x_train = np.load("data/data_orig/img_uint8.npy")
y_train = np.load("data/data_orig/msk_uint8.npy")

# Verify the mask shape and values
print("y_train uniq (must only be 0 and 255):", np.unique(y_train))
print("x_train shape:", x_train.shape, "y_train_shape:", y_train.shape)

# Total number of images
num_images = x_train.shape[0]

# Data generator for training and validation
data_gen_args = dict(
    rescale=1.0 / 255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

data_mean, data_std = calc_mean_std(x_train)

image_datagen = ImageDataGenerator(
    **data_gen_args,
    preprocessing_function=partial(normalize_batch, mean=data_mean, std=data_std),
)
mask_datagen = ImageDataGenerator(
    **data_gen_args,
    preprocessing_function=partial(normalize_batch, mean=data_mean, std=data_std),
)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
batch_sz = BATCH_SIZE

# Train-val split
num_train = int(num_images * TRAIN_FRACTION)
num_val = int(num_images * VAL_FRACTION)

# train val image and mask generators
train_image_generator = image_datagen.flow(
    x_train, batch_size=batch_sz, shuffle=True, subset="training", seed=seed
)
train_mask_generator = mask_datagen.flow(
    y_train, batch_size=batch_sz, shuffle=True, subset="training", seed=seed
)
val_image_generator = image_datagen.flow(
    x_train, batch_size=batch_sz, shuffle=True, subset="validation", seed=seed
)
val_mask_generator = mask_datagen.flow(
    y_train, batch_size=batch_sz, shuffle=True, subset="validation", seed=seed
)

# combine generators into one which yields image and masks
train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)

model = get_mobile_unet_mnetv2(
    finetune=FINETUNE,
    model_type=MODEL_TYPE,
    pretrain_model_path=PRETRAINED_MODEL_PATH,
    quant_aware_train=Q_AWARE_TRAIN,
)

# print model summary
model.summary()

# save checkpoints
checkpoint = ModelCheckpoint(
    CHECKPOINT_FMT,
    monitor="val_loss",
    verbose=1,
    save_weights_only=False,
    save_best_only=True,
    mode="min",
)

# Callbacks
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=15, min_lr=0.000001, verbose=1)
tensorboard = TensorBoard(
    log_dir=LOG_PATH, histogram_freq=0, write_graph=True, write_images=True
)
callbacks_list = [checkpoint, tensorboard, reduce_lr]

# compile model
model.compile(
    loss="binary_crossentropy",
    optimizer=OPTIMIZER(**OPTIM_PARAMS),
    metrics=["accuracy"],
)

# Train the model
model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=num_train / batch_sz,
    validation_data=val_generator,
    validation_steps=num_val / batch_sz,
    use_multiprocessing=True,
    workers=2,
    callbacks=callbacks_list,
)

if Q_AWARE_TRAIN:
    from modules.mobileunet_mobilenet_v2.munet_mnetv2 import NoOpQuantizeConfig, tfmot

    with custom_object_scope({"NoOpQuantizeConfig": NoOpQuantizeConfig}):
        q_aware_model = tfmot.quantization.keras.quantize_model(model)

        # `quantize_model` requires a recompile.
        q_aware_model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=1e-3),
            metrics=["accuracy"],
        )

        q_aware_model.summary()
        # better to use a subset of the original training data for fitting here
        q_aware_model.fit_generator(
            train_generator,
            epochs=30,
            steps_per_epoch=num_train / batch_sz,
            validation_data=val_generator,
            validation_steps=num_val / batch_sz,
            use_multiprocessing=True,
            workers=2,
            callbacks=callbacks_list,
        )

        converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        quantized_tflite_model = converter.convert()
        # save converted quantization model to tflite format
        with open("quantized.tflite", "wb") as tf_ptr:
            tf_ptr.write(quantized_tflite_model)
