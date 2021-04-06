import numpy as np
from functools import partial

# Import libraries
from tensorflow.python.client import device_lib
device_list = [dev.device_type for dev in device_lib.list_local_devices()]

if 'GPU' in device_list:
    import os
    # if tensorflow gpu exists
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    import tensorflow.keras.backend as K
    from tensorflow.keras.utils import plot_model
    from tensorflow.keras.optimizers import SGD, Adam
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.layers import UpSampling2D, Conv2DTranspose, BatchNormalization, Dropout
    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback, ReduceLROnPlateau
    from tensorflow.keras.layers import Dense, Input, Flatten, concatenate, Reshape, Conv2D, MaxPooling2D, Lambda, Activation, Conv2DTranspose
else:
    import keras
    # if tensorflow cpu
    import tensorflow as tf
    import keras.backend as K
    from keras.utils import plot_model
    from keras.optimizers import SGD, Adam
    from keras.models import Model, load_model
    from keras.applications.mobilenet_v2 import MobileNetV2
    from keras.preprocessing.image import ImageDataGenerator
    from keras.layers import UpSampling2D, Conv2DTranspose, BatchNormalization, Dropout
    from keras.callbacks import TensorBoard, ModelCheckpoint, Callback, ReduceLROnPlateau
    from keras.layers import Dense, Input, Flatten, concatenate, Reshape, Conv2D, MaxPooling2D, Lambda, Activation, Conv2DTranspose


# Set bilinear or transpose conv model
MODEL_TYPE = "transpose"  # "bilinear"
if MODEL_TYPE not in {"transpose", "bilinear"}:
    raise AttributeError("MODEL_TYPE must be transpose or bilinear")

# Load the dataset
x_train = np.load("data/voc_img_uint8.npy")
y_train = np.load("data/voc_msk_uint8.npy")

# Configure save paths and batch size
PRETRAINED = 'checkpoints/deconv_bnoptimized_munet.h5'
CHECKPOINT = "checkpoints/deconv_bnoptimized_munet-{epoch:02d}-{val_loss:.2f}.hdf5"
LOGS = './logs'
BATCH_SIZE = 32

# Verify the mask shape and values
print("y_train uniq (must only be 0 and 255) = ", np.unique(y_train))
print("x_train shape=", x_train.shape, "y_train_shape", y_train.shape)

# Total number of images
num_images = x_train.shape[0]

# Preprocessing function (runtime)


def calc_mean_std(img_arr):
    """img_arr must have shape [b_size, width, height, channel]
    """
    nimages, mean, std = 0, 0., 0.
    # Rearrange batch to be the shape of [B, C, W * H]
    img_arr = img_arr.reshape(img_arr.shape[0], img_arr.shape[-1], -1)
    # Compute mean and std here
    mean = img_arr.mean(2).sum(0) / img_arr.shape[0]
    std = img_arr.std(2).sum(0) / img_arr.shape[0]

    return mean / 255, std / 255


def normalize_batch(imgs, mean=[0.50693673, 0.47721124, 0.44640532], std=[0.28926975, 0.27801928, 0.28596011]):
    if imgs.shape[-1] > 1:
        return (imgs - np.array(mean)) / np.array(std)
    else:
        return imgs.round()


def denormalize_batch(imgs, should_clip=True, mean=[0.50693673, 0.47721124, 0.44640532], std=[0.28926975, 0.27801928, 0.28596011]):
    imgs = (imgs * np.array(mean)
            ) + np.array(std)

    if should_clip:
        imgs = np.clip(imgs, 0, 1)
    return imgs


# Data generator for training and validation
data_gen_args = dict(rescale=1. / 255,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     validation_split=0.2
                     )

data_mean, data_std = calc_mean_std(x_train)

image_datagen = ImageDataGenerator(
    **data_gen_args, preprocessing_function=partial(normalize_batch, mean=data_mean, std=data_std))
mask_datagen = ImageDataGenerator(
    **data_gen_args,  preprocessing_function=partial(normalize_batch, mean=data_mean, std=data_std))

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
batch_sz = BATCH_SIZE

# Train-val split (80-20)
num_train = int(num_images * 0.8)
num_val = int(num_images * 0.2)

train_image_generator = image_datagen.flow(
    x_train,
    batch_size=batch_sz,
    shuffle=True,
    subset='training',
    seed=seed)

train_mask_generator = mask_datagen.flow(
    y_train,
    batch_size=batch_sz,
    shuffle=True,
    subset='training',
    seed=seed)

val_image_generator = image_datagen.flow(
    x_train,
    batch_size=batch_sz,
    shuffle=True,
    subset='validation',
    seed=seed)

val_mask_generator = mask_datagen.flow(
    y_train,
    batch_size=batch_sz,
    shuffle=True,
    subset='validation',
    seed=seed)


# combine generators into one which yields image and masks
train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)


# Convolution block with Transpose Convolution
def deconv_block(tensor, nfilters, size=3, padding='same', kernel_initializer='he_normal'):
    y = Conv2DTranspose(filters=nfilters, kernel_size=size, strides=2,
                        padding=padding, kernel_initializer=kernel_initializer)(tensor)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    y = Activation("relu")(y)

    return y

# Convolution block with Upsampling+Conv2D


def deconv_block_rez(tensor, nfilters, size=3, padding='same', kernel_initializer='he_normal'):
    y = UpSampling2D(size=(2, 2), interpolation='bilinear')(tensor)
    y = Conv2D(filters=nfilters, kernel_size=(size, size),
               padding='same', kernel_initializer=kernel_initializer)(y)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    y = Activation("relu")(y)

    return y

# Model architecture


def get_mobile_unet(finetune=False, pretrained=False):

    # Load pretrained model (if any)
    if (pretrained):
        model = load_model(PRETRAINED)
        print("Loaded pretrained model ...\n")

        if finetune:
            print("Freezing initial layer ...\n")
            # freezing initial layers adopted from mobilenet v2
            for layer in model.layers[:101]:
                layer.trainable = False
        return model

    # Encoder/Feature extractor
    mnv2 = MobileNetV2(input_shape=(
        128, 128, 3), alpha=0.5, include_top=False, weights='imagenet')

    if (finetune):
        print("Freezing initial layer ...\n")
        for layer in mnv2.layers[:-3]:
            layer.trainable = False

    x = mnv2.layers[-4].output

    if MODEL_TYPE == "transpose":
        deconv = deconv_block
    elif MODEL_TYPE == "bilinear":
        deconv = deconv_block_rez

    # Decoder
    x = deconv(x, 512)
    x = concatenate([x, mnv2.get_layer('block_13_expand_relu').output], axis=3)

    x = deconv(x, 256)
    x = concatenate([x, mnv2.get_layer('block_6_expand_relu').output], axis=3)

    x = deconv(x, 128)
    x = concatenate([x, mnv2.get_layer('block_3_expand_relu').output], axis=3)

    x = deconv(x, 64)
    x = concatenate([x, mnv2.get_layer('block_1_expand_relu').output], axis=3)

    if MODEL_TYPE == "transpose":
        x = Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                            padding='same', kernel_initializer='he_normal')(x)
    elif MODEL_TYPE == "bilinear":
        x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        x = Conv2D(filters=32, kernel_size=3, padding='same',
                   kernel_initializer='he_normal')(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(1, (1, 1), padding='same')(x)
    x = Activation('sigmoid', name="op")(x)

    model = Model(inputs=mnv2.input, outputs=x)
    return model


model = get_mobile_unet(finetune=False, pretrained=True)

# Model summary
# model.summary()

# Plot model architecture
# plot_model(model, to_file='portrait_seg.png')

# Save checkpoints
checkpoint = ModelCheckpoint(CHECKPOINT, monitor='val_loss', verbose=1,
                             save_weights_only=False, save_best_only=True, mode='min')

# Callbacks
reduce_lr = ReduceLROnPlateau(
    factor=0.5, patience=15, min_lr=0.000001, verbose=1)
tensorboard = TensorBoard(log_dir=LOGS, histogram_freq=0,
                          write_graph=True, write_images=True)

callbacks_list = [checkpoint, tensorboard, reduce_lr]

# compile model
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-3), metrics=['accuracy'])

# Train the model
model.fit_generator(
    train_generator,
    epochs=300,
    steps_per_epoch=num_train / batch_sz,
    validation_data=val_generator,
    validation_steps=num_val / batch_sz,
    use_multiprocessing=True,
    workers=2,
    callbacks=callbacks_list)

# Sample run: python train.py
