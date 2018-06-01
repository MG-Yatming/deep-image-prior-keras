from keras.layers import Conv2D, BatchNormalization, ZeroPadding2D, UpSampling2D, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.layers import concatenate
from model.layer import ReflectPadding2D, BilinearUpsampling2D


def padding(x, size, pad_mode):
    if size == 0:
        return x
    if pad_mode == 'reflection':
        x = ReflectPadding2D(x, size)
    elif pad_mode == 'zero':
        x = ZeroPadding2D(size)(x)
    return x


def upsampling(x, size, upsample_mode):
    if upsample_mode == 'nearest':
        x = UpSampling2D(size)(x)
    elif upsample_mode == 'bilinear':
        x = BilinearUpsampling2D(x, size)
    return x


def d(x, n, k, pad_mode, use_bias=True):
    pad_size = int((k - 1) / 2)
    x = padding(x, pad_size, pad_mode)
    x = Conv2D(n, k, strides=2, use_bias=use_bias)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = padding(x, pad_size, pad_mode)
    x = Conv2D(n, k, strides=1, use_bias=use_bias)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    return x


def s(x, n, k, pad_mode, use_bias=True):
    pad_size = int((k - 1) / 2)
    x = padding(x, pad_size, pad_mode)
    x = Conv2D(n, k, strides=1, use_bias=use_bias)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    return x


def u(x, n, k, pad_mode, upsample_mode, use_bias=True):
    x = BatchNormalization()(x)

    pad_size = int((k - 1) / 2)
    x = padding(x, pad_size, pad_mode)
    x = Conv2D(n, k, strides=1, use_bias=use_bias)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(n, 1, strides=1, use_bias=use_bias)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = upsampling(x, 2, upsample_mode)

    return x


def Hourglass(input_size,
              input_channel=2,
              output_channel=3,
              num_up=[16, 32, 64, 128, 128],
              num_down=[16, 32, 64, 128, 128],
              num_skip=[4, 4, 4, 4, 4],
              k_up=[3, 3, 3, 3, 3],
              k_down=[3, 3, 3, 3, 3],
              k_skip=[1, 1, 1, 1, 1],
              pad_mode='reflection',
              upsample_mode='nearest',
              use_bias=True
              ):
    w, h = input_size
    input = Input((w, h, input_channel))
    x = input

    n = len(num_up)

    stack = []
    for i in range(n - 1):
        x = d(x, num_down[i], k_down[i], pad_mode, use_bias)
        if num_skip[i] == 0:
            stack.append(None)
        else:
            skip = s(x, num_skip[i], k_skip[i], pad_mode, use_bias)
            stack.append(skip)

    # bottleneck
    x = d(x, num_down[n - 1], k_down[n - 1], pad_mode, use_bias)
    x = s(x, num_skip[n - 1], k_skip[n - 1], pad_mode, use_bias)
    x = u(x, num_up[n - 1], k_up[n - 1], pad_mode, upsample_mode, use_bias)

    for i in range(n - 1):
        j = n - i - 2
        if stack[j] != None:
            x = concatenate([x, stack[j]], axis=-1)
        x = u(x, num_up[j], k_up[j], pad_mode, upsample_mode, use_bias)

    output = Conv2D(output_channel, 1, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output, name='hourglass')
    return model
