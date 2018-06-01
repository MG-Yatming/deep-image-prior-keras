from utils import *
import os
from model.hourglass import Hourglass
from keras.optimizers import Adam
from keras.metrics import mse
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

### gpu options
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

### environment options
exp_name = 'denoising'
data_path = 'data/%s/' % exp_name
# image_path = 'snail.jpg'
image_path = 'F16_GT.png'
save_path = 'result/%s/%s/' % (exp_name, image_path.split('.')[0])

if not os.path.exists('result/%s/' % exp_name):
    os.mkdir('result/%s/' % exp_name)
if not os.path.exists(save_path):
    os.mkdir(save_path)

### experiment options
sigma = 1. / 30
method = 'random'
input_channel = 3
lr = 0.01

### load data
img = load_data(data_path + image_path)
img = crop_image(img)
if image_path == 'snail.jpg':
    num_iter = 10000
    img.save(save_path + 'ori.png')
    x = preprocess(img)
elif image_path == 'F16_GT.png':
    num_iter = 5000
    x = preprocess(img)
    x = get_noisy_image(x, 25 / 255.)
    postprocess(x[0]).save(save_path + 'ori.png')

### build code z
b, w, h, c = x.shape

z = make_noise(method, input_channel, (w, h))

### build model
if image_path == 'snail.jpg':
    model = Hourglass((w, h), input_channel, c,
                      num_up=[16, 32, 64, 128, 128],
                      num_down=[16, 32, 64, 128, 128],
                      num_skip=[0, 0, 4, 4, 128],
                      k_up=[3, 3, 3, 3, 3],
                      k_down=[3, 3, 3, 3, 3],
                      k_skip=[0, 0, 1, 1, 1],
                      upsample_mode='bilinear'
                      )
elif image_path == 'F16_GT.png':
    model = Hourglass((w, h), input_channel, c,
                      num_up=[128, 128, 128, 128, 128],
                      num_down=[128, 128, 128, 128, 128],
                      num_skip=[4, 4, 4, 4, 128],
                      k_up=[3, 3, 3, 3, 3],
                      k_down=[3, 3, 3, 3, 3],
                      k_skip=[1, 1, 1, 1, 1],
                      upsample_mode='bilinear'
                      )

model.compile(optimizer=Adam(lr=lr), loss=mse)
model.summary()

losses = []
for i in range(num_iter+1):
    loss = model.train_on_batch(add_noise(z, sigma), x)
    losses.append(loss)

    if i % 100 == 0:
        print('iter %d loss %f' % (i, loss))
        y = model.predict_on_batch(z)
        postprocess(y[0]).save(save_path + '%d.png' % i)

model.save(save_path + 'model.h5')

del losses[-1]
plt.plot(list(range(num_iter)), losses)
plt.savefig(save_path + 'loss.png')