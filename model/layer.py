import tensorflow as tf
from keras.layers import Lambda
import numpy as np
from keras.layers import Conv2D
from model.ops import lanczos2_kernel


def ReflectPadding2D(x, padding=1, name=None):
    pad = lambda x: tf.pad(x, paddings=[[0, 0], [padding, padding], [padding, padding], [0, 0]], mode='REFLECT')
    return Lambda(pad, name=name)(x)


def BilinearUpsampling2D(x, size=2, name=None):
    shape = x.get_shape()
    w = int(shape[1])
    h = int(shape[2])
    new_w = int(round(w * size))
    new_h = int(round(h * size))
    resized = lambda x: tf.image.resize_images(x, [new_w, new_h], method=tf.image.ResizeMethod.BILINEAR)
    return Lambda(resized, name=name)(x)


def Lanczos2Conv2D(x, channel, factor=4, name=None):
    kernel = lanczos2_kernel(factor)
    weights = np.zeros((kernel.shape[0], kernel.shape[1], channel, channel))
    for i in range(channel):
        weights[:, :, i, i] = kernel
    x = ReflectPadding2D(x, int((kernel.shape[0] - 1) / 2.))
    downsampling = Conv2D(channel, kernel.shape[0], strides=factor, use_bias=False)
    x = downsampling(x)
    downsampling.set_weights([weights])
    downsampling.trainable = False
    return x
