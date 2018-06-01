from keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np
from PIL import Image


def load_data(path):
    img = load_img(path)
    return img


def preprocess(img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')
    img = img / 255
    return img


def postprocess(img):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img = array_to_img(img)
    return img


def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
        int((img.size[0] - new_size[0]) / 2),
        int((img.size[1] - new_size[1]) / 2),
        int((img.size[0] + new_size[0]) / 2),
        int((img.size[1] + new_size[1]) / 2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped


def get_noisy_image(img, sigma):
    """Adds Gaussian noise to an image."""
    img_noisy = np.clip(img + np.random.normal(scale=sigma, size=img.shape), 0, 1).astype(np.float32)
    return img_noisy


def make_noise(method, channel, sizes):
    if method == 'random':
        shape = (1, sizes[0], sizes[1], channel)
        noise = np.random.uniform(0, 0.1, size=shape)
    elif method == 'meshgrid':
        X, Y = np.meshgrid(np.arange(0, sizes[1]) / float(sizes[1] - 1),
                           np.arange(0, sizes[0]) / float(sizes[0] - 1))
        X = np.expand_dims(X, axis=-1)
        Y = np.expand_dims(Y, axis=-1)
        noise = np.concatenate([X, Y], axis=-1)
        noise = np.expand_dims(noise, axis=0)
    return noise


def add_noise(x, sigma):
    noise = np.random.normal(0, sigma, size=x.shape)
    return x + noise


def low_resolution(img, factor):
    lr_size = [
        img.size[0] // factor,
        img.size[1] // factor
    ]

    img = img.resize(lr_size, Image.ANTIALIAS)
    return img


