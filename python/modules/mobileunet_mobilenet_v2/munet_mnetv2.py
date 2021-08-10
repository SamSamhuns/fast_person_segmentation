from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Concatenate, Conv2D, Activation
from tensorflow.keras.layers import UpSampling2D, Conv2DTranspose, BatchNormalization, Dropout

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras import quantize_config


class NoOpQuantizeConfig(quantize_config.QuantizeConfig):
    """QuantizeConfig which does not quantize any part of the layer."""

    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}


def get_quant_aware_concatenation_layer():
    # set a no-quantize operation for concatenate layer with axis = 3
    Concatenate_quant = tfmot.quantization.keras.quantize_annotate_layer(
        Concatenate(axis=3), quantize_config=NoOpQuantizeConfig())
    return Concatenate_quant


def deconv_block(tensor, nfilters, size=3, padding='same', kernel_initializer='he_normal'):
    """
    Convolution block with Transpose Convolution
    """
    y = Conv2DTranspose(filters=nfilters, kernel_size=size, strides=2,
                        padding=padding, kernel_initializer=kernel_initializer)(tensor)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    y = Activation("relu")(y)

    return y


def deconv_block_rez(tensor, nfilters, size=3, padding='same', kernel_initializer='he_normal'):
    """
    Convolution block with Upsampling+Conv2D
    """
    y = UpSampling2D(size=(2, 2), interpolation='bilinear')(tensor)
    y = Conv2D(filters=nfilters, kernel_size=(size, size),
               padding='same', kernel_initializer=kernel_initializer)(y)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    y = Activation("relu")(y)

    return y


def get_mobile_unet_mnetv2(finetune=False, model_type="transpose", pretrain_model_path=None, quant_aware_train=False):
    # Model architecture
    # Load pretrained model
    if pretrain_model_path is not None:
        model = load_model(pretrain_model_path)
        print(f"Loaded pretrained mode {pretrain_model_path}\n")

        if finetune:
            print("Freezing initial layers for finetuning\n")
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

    if model_type == "transpose":
        deconv = deconv_block
    elif model_type == "bilinear":
        deconv = deconv_block_rez

    # set right concatenate for quant ware train if used
    if quant_aware_train:
        print("Using quant aware Concatenate Layer")
        ConcatenateLayer = get_quant_aware_concatenation_layer()
    else:
        ConcatenateLayer = Concatenate(axis=3)

    # Decoder
    # concatenates that do not use quant config
    x = deconv(x, 512)
    x = ConcatenateLayer([x, mnv2.get_layer('block_13_expand_relu').output])

    x = deconv(x, 256)
    x = ConcatenateLayer([x, mnv2.get_layer('block_6_expand_relu').output])

    x = deconv(x, 128)
    x = ConcatenateLayer([x, mnv2.get_layer('block_3_expand_relu').output])

    x = deconv(x, 64)
    x = ConcatenateLayer([x, mnv2.get_layer('block_1_expand_relu').output])

    if model_type == "transpose":
        x = Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                            padding='same', kernel_initializer='he_normal')(x)
    elif model_type == "bilinear":
        x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        x = Conv2D(filters=32, kernel_size=3, padding='same',
                   kernel_initializer='he_normal')(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(1, (1, 1), padding='same')(x)
    x = Activation('sigmoid', name="op")(x)

    model = Model(inputs=mnv2.input, outputs=x)
    return model
