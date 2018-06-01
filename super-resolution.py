from utils import *
import os
from model.hourglass import Hourglass
from keras.optimizers import Adam
from keras.metrics import mse
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import PIL
from keras.models import Model
from model.layer import Lanczos2Conv2D

### gpu options
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

### experiment options
method = 'random'
input_channel = 32
lr = 0.01

factor = 8
if factor == 4:
    num_iter = 4000
    sigma = 1. / 30
elif factor == 8:
    num_iter = 6000
    sigma = 1. / 20

### environment options
exp_name = 'sr'
data_path = 'data/%s/' % exp_name
# image_path = 'snail.jpg'
image_path = 'zebra_GT.png'
save_path = 'result/%s/%s_%dx/' % (exp_name, image_path.split('.')[0], factor)

if not os.path.exists('result/%s/' % exp_name):
    os.mkdir('result/%s/' % exp_name)
if not os.path.exists(save_path):
    os.mkdir(save_path)

### load data
hr_img = load_data(data_path + image_path)
hr_img = crop_image(hr_img)
lr_img = low_resolution(hr_img, factor)
hr_x = preprocess(hr_img)
lr_x = preprocess(lr_img)
postprocess(hr_x[0]).save(save_path + 'hr_ori.png')
postprocess(lr_x[0]).save(save_path + 'lr_ori.png')

### baseline method
### bicubic, sharpened bicubic and nearest
lr_img.resize(hr_img.size, Image.BICUBIC).save(save_path + 'hr_bicubic.png')
lr_img.resize(hr_img.size, Image.NEAREST).save(save_path + 'hr_nearest.png')
lr_img.resize(hr_img.size, Image.BICUBIC).filter(PIL.ImageFilter.UnsharpMask()).save(save_path + 'hr_sharpened.png')

### build code z
b, w, h, c = hr_x.shape

z = make_noise(method, input_channel, (w, h))

### build model
g = Hourglass((w, h), input_channel, c,
                  num_up=[128, 128, 128, 128, 128],
                  num_down=[128, 128, 128, 128, 128],
                  num_skip=[4, 4, 4, 4, 128],
                  k_up=[3, 3, 3, 3, 3],
                  k_down=[3, 3, 3, 3, 3],
                  k_skip=[1, 1, 1, 1, 1],
                  upsample_mode='bilinear'
                  )

input = g.input
x = g.output
output = Lanczos2Conv2D(x, c, factor)
model = Model(inputs=input, outputs=output, name='g_trainer')

model.compile(optimizer=Adam(lr=lr), loss=mse)
model.summary()

losses = []
for i in range(num_iter + 1):
    loss = model.train_on_batch(add_noise(z, sigma), lr_x)
    losses.append(loss)

    if i % 100 == 0:
        print('iter %d loss %f' % (i, loss))
        y = g.predict_on_batch(z)
        postprocess(y[0]).save(save_path + '%d.png' % i)

model.save(save_path + 'model.h5')

del losses[-1]
plt.plot(list(range(num_iter)), losses)
plt.savefig(save_path + 'loss.png')
