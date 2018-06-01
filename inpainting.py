from utils import *
import os
from model.hourglass import Hourglass
from keras.optimizers import Adam
from keras.metrics import mse
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import multiply, Input
from keras.models import Model

### gpu options
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

### environment options
exp_name = 'inpainting'
data_path = 'data/%s/' % exp_name
# image_path = 'kate.png'
image_path = 'library.png'
# image_path = 'vase.png'
save_path = 'result/%s/%s/' % (exp_name, image_path.split('.')[0])

if not os.path.exists('result/%s/' % exp_name):
    os.mkdir('result/%s/' % exp_name)
if not os.path.exists(save_path):
    os.mkdir(save_path)

### load data
img = load_data(data_path + image_path)
mask = load_data(data_path + image_path.replace('.', '_mask.'))
img = crop_image(img)
mask = crop_image(mask)
img.save(save_path + 'img_ori.png')
mask.save(save_path + 'mask_ori.png')
img = preprocess(img)
mask = preprocess(mask)
miss = img * mask
postprocess(miss[0]).save(save_path + 'miss_ori.png')

### build code z
b, w, h, c = img.shape
### experiment options
### build model
if image_path == 'kate.png':
    sigma = 1. / 30
    method = 'random'
    input_channel = 32
    lr = 0.01
    num_iter = 6000

    z = make_noise(method, input_channel, (w, h))

    g = Hourglass((w, h), input_channel, c,
                      num_up=[128, 128, 128, 128, 128],
                      num_down=[128, 128, 128, 128, 128],
                      num_skip=[4, 4, 4, 4, 128],
                      k_up=[3, 3, 3, 3, 3],
                      k_down=[3, 3, 3, 3, 3],
                      k_skip=[1, 1, 1, 1, 1],
                      upsample_mode='bilinear'
                      )
elif image_path == 'library.png':
    sigma = 0.
    method = 'random'
    input_channel = 32
    lr = 0.1
    num_iter = 5000

    z = make_noise(method, input_channel, (w, h))

    g = Hourglass((w, h), input_channel, c,
                      num_up=[16, 32, 64, 128, 128],
                      num_down=[16, 32, 64, 128, 128],
                      num_skip=[0, 0, 0, 4, 128],
                      k_up=[5, 5, 5, 5, 5],
                      k_down=[3, 3, 3, 3, 3],
                      k_skip=[0, 0, 0, 1, 1],
                      upsample_mode='nearest'
                      )
elif image_path == 'vase.png':
    sigma = 1. / 30
    method = 'meshgrid'
    input_channel = 2
    lr = 0.01
    num_iter = 5000

    z = make_noise(method, input_channel, (w, h))

    g = Hourglass((w, h), input_channel, c,
                      num_up=[128, 128, 128, 128, 128],
                      num_down=[128, 128, 128, 128, 128],
                      num_skip=[0, 0, 0, 4, 128],
                      k_up=[3, 3, 3, 3, 3],
                      k_down=[3, 3, 3, 3, 3],
                      k_skip=[0, 0, 0, 1, 1],
                      upsample_mode='bilinear'
                      )
input = g.input
mask_input = Input((w, h, c))
x = g.output
output = multiply([x, mask_input])
model = Model(inputs=[input, mask_input], outputs=output, name='g_trainer')

model.compile(optimizer=Adam(lr=lr), loss=mse)
model.summary()

losses = []
for i in range(num_iter + 1):
    loss = model.train_on_batch([add_noise(z, sigma), mask], miss)
    losses.append(loss)

    if i % 100 == 0:
        print('iter %d loss %f' % (i, loss))
        y = g.predict_on_batch(z)
        postprocess(y[0]).save(save_path + '%d.png' % i)

model.save(save_path + 'model.h5')

del losses[-1]
plt.plot(list(range(num_iter)), losses)
plt.savefig(save_path + 'loss.png')
