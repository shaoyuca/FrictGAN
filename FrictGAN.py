import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
import datetime
import os
import time
from matplotlib import pyplot as plt
from IPython import display
import numpy as np
import cv2
import librosa.display
from scipy import signal


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

EPOCHS = 400
log_dir = "logs/"
PATH = "dataset/"
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Training
classes = 400
f1 = [os.path.join(PATH, 'train/2/image/', "%d.jpg"%i)for i in range (200, classes+200)]
f2 = [os.path.join(PATH, 'train/3/image/', "%d.jpg"%i)for i in range (150, classes+150)]
f3 = [os.path.join(PATH, 'train/4/image/', "%d.jpg"%i)for i in range (100, classes+100)]
f4 = [os.path.join(PATH, 'train/5/image/', "%d.jpg"%i)for i in range (200, classes+200)]
f5 = [os.path.join(PATH, 'train/8/image/', "%d.jpg"%i)for i in range (200, classes+200)]
filepaths = f1+f2+f3+f4+f5

f_1 = [os.path.join(PATH, 'train/2/output_data/', "%d.txt"%i)for i in range (200, classes+200)]
f_2 = [os.path.join(PATH, 'train/3/output_data/', "%d.txt"%i)for i in range (150, classes+150)]
f_3 = [os.path.join(PATH, 'train/4/output_data/', "%d.txt"%i)for i in range (100, classes+100)]
f_4 = [os.path.join(PATH, 'train/5/output_data/', "%d.txt"%i)for i in range (200, classes+200)]
f_5 = [os.path.join(PATH, 'train/8/output_data/', "%d.txt"%i)for i in range (200, classes+200)]
filepaths_1 = f_1+f_2+f_3+f_4+f_5

# testing & validing
f6 = [os.path.join(PATH, 'test/2/image/', "%d.jpg"%j)for j in range (100, 200)]
f7 = [os.path.join(PATH, 'test/3/image/', "%d.jpg"%j)for j in range (100, 150)]
f7a = [os.path.join(PATH, 'test/3/image/', "%d.jpg"%j)for j in range (550, 600)]
f8 = [os.path.join(PATH, 'test/4/image/', "%d.jpg"%j)for j in range (500, 600)]
f9 = [os.path.join(PATH, 'test/5/image/', "%d.jpg"%j)for j in range (100, 200)]
f10 = [os.path.join(PATH, 'test/8/image/', "%d.jpg"%j)for j in range (100, 200)]
Afilepaths = f6+f7+f7a+f8+f9+f10

##########################################
m_1 = [os.path.join(PATH, 'valid/2/image/', "%d.jpg"%j)for j in range (600, 601)]
m_2 = [os.path.join(PATH, 'valid/3/image/', "%d.jpg"%j)for j in range (600, 601)]
m_3 = [os.path.join(PATH, 'valid/4/image/', "%d.jpg"%j)for j in range (600, 601)]
m_4 = [os.path.join(PATH, 'valid/5/image/', "%d.jpg"%j)for j in range (600, 601)]
m_5 = [os.path.join(PATH, 'valid/8/image/', "%d.jpg"%j)for j in range (600, 601)]

filepaths_2 = m_1+m_2+m_3+m_4+m_5

f_6 = [os.path.join(PATH, 'test/2/output_data/', "%d.txt"%j)for j in range (100, 200)]
f_7 = [os.path.join(PATH, 'test/3/output_data/', "%d.txt"%j)for j in range (100, 150)]
f_7a = [os.path.join(PATH, 'test/3/output_data/', "%d.txt"%j)for j in range (550, 600)]
f_8 = [os.path.join(PATH, 'test/4/output_data/', "%d.txt"%j)for j in range (500, 600)]
f_9 = [os.path.join(PATH, 'test/5/output_data/', "%d.txt"%j)for j in range (100, 200)]
f_10 = [os.path.join(PATH, 'test/8/output_data/', "%d.txt"%j)for j in range (100, 200)]
Bfilepaths_1 = f_6+f_7+f_7a+f_8+f_9+f_10

m_6 = [os.path.join(PATH, 'valid/2/output_data/', "%d.txt"%j)for j in range (600, 601)]
m_7 = [os.path.join(PATH, 'valid/3/output_data/', "%d.txt"%j)for j in range (600, 601)]
m_8 = [os.path.join(PATH, 'valid/4/output_data/', "%d.txt"%j)for j in range (600, 601)]
m_9 = [os.path.join(PATH, 'valid/5/output_data/', "%d.txt"%j)for j in range (600, 601)]
m_10 = [os.path.join(PATH, 'valid/8/output_data/', "%d.txt"%j)for j in range (600, 601)]
Bfilepaths_2 = m_6+m_7+m_8+m_9+m_10


def my_func(txt):
    a = np.loadtxt(txt.decode())
    return a.astype(np.float32)

def my_func_1(image):
    b = tf.compat.v1.keras.utils.normalize(image)
    return b.astype(np.float32)

def load(image_name, txt_name):
    # load image data
    image = tf.io.read_file(image_name)
    image = tf.image.decode_jpeg(image)

    # load spectrogram data
    txt = tf.compat.v1.py_func(my_func, [txt_name], tf.float32)
    txt = tf.reshape(txt, [257, 11, 1])

    input_image = tf.cast(image, tf.float32)
    real_data = tf.cast(txt, tf.float32)

    input_image = tf.image.random_contrast(input_image, lower=4, upper=6)
    input_image = tf.image.random_saturation(input_image, lower=2, upper=4)
    #input_image = tf.compat.v1.py_func(my_func_1, [input_image], tf.float32)

    input_image = (input_image / 127.5) - 1

    return input_image, real_data

BUFFER_SIZE = 200
BATCH_SIZE = 2

train_image = tf.constant(filepaths)
train_spec = tf.constant(filepaths_1)
train_dataset = tf.data.Dataset.from_tensor_slices((train_image, train_spec))
train_dataset = train_dataset.map(load, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)


image_name_1 = tf.constant(Afilepaths)
txt_name_1 = tf.constant(Bfilepaths_1)
test_dataset = tf.data.Dataset.from_tensor_slices((image_name_1, txt_name_1))
test_dataset = test_dataset.map(load, num_parallel_calls=tf.data.experimental.AUTOTUNE)
show_dataset = test_dataset.shuffle(10).batch(1)
test_dataset = test_dataset.batch(1)


valid_image = tf.constant(filepaths_2)
valid_txt = tf.constant(Bfilepaths_2)
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_image, valid_txt))
valid_dataset = valid_dataset.map(load, num_parallel_calls=tf.data.experimental.AUTOTUNE)
showtrain_dataset = valid_dataset.shuffle(10).batch(1)
valid_dataset = valid_dataset.batch(1)

# sample image from valid dataset
sample_trainimage, sample_trainspec = next(iter(showtrain_dataset))
# sample image from test dataset
sample_image, sample_spec = next(iter(show_dataset))

#---------------------------------------------------------------------------------------
# build generator
def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return tf.keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return tf.keras.layers.LayerNormalization

def Generator():
    concat_axis = 3
    norm = 'batch_norm'
    Norm = _get_norm_layer(norm)
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02, seed=None)
    a = inputs = keras.Input(shape=(1024, 1024, 3))

    conv0_1 = tf.keras.layers.Conv2D(32, (4, 4), strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(a)
    #conv0_2 = Norm()(conv0_1)
    conv0_3 = tf.nn.relu(conv0_1)

    conv1_1 = tf.keras.layers.Conv2D(64, (4, 4), strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(conv0_3)
    conv1_2 = Norm()(conv1_1)
    conv1_3 = tf.nn.relu(conv1_2)

    conv2_1 = tf.keras.layers.Conv2D(128, (4, 4), strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(conv1_3)
    conv2_2 = Norm()(conv2_1)
    conv2_3 = tf.nn.relu(conv2_2)

    conv3_1 = tf.keras.layers.Conv2D(256, (4, 4), strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(conv2_3)
    conv3_2 = Norm()(conv3_1)
    conv3_3 = tf.nn.relu(conv3_2)

    conv4_1 = tf.keras.layers.Conv2D(512, (4, 4), strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(conv3_3)
    conv4_2 = Norm()(conv4_1)
    conv4_3 = tf.nn.relu(conv4_2)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4_3)

    conv5_1 = tf.keras.layers.Conv2D(512, (4, 4), strides=4, padding='same', kernel_initializer=initializer, use_bias=False)(drop4)
    conv5_2 = Norm()(conv5_1)
    conv5_3 = tf.nn.relu(conv5_2)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5_3)

    conv5_4 = tf.keras.layers.Conv2D(512, (4, 4), strides=4, padding='same', kernel_initializer=initializer, use_bias=False)(drop5)
    conv5_5 = Norm()(conv5_4)
    conv5_6 = tf.nn.relu(conv5_5)
    drop6 = tf.keras.layers.Dropout(0.5)(conv5_6)


    conv6_1 = tf.keras.layers.Conv2DTranspose(512, (4, 4), strides=(4, 2), padding='same', kernel_initializer=initializer, use_bias=False)(drop6)  # 8, 4
    crop_6 = tf.keras.layers.Cropping2D(cropping=((0, 0), (2, 2)), name='cropped_conv5_2')(conv5_1) # 8 - 2
    up6 = tf.keras.layers.concatenate([conv6_1, crop_6], axis=concat_axis, name='skip_connection1')
    up6_1 = tf.keras.layers.Conv2D(512, (2, 2), padding='valid', kernel_initializer=initializer, name='conv8_1')(up6) # 7 - 3
    up6_2 = tf.keras.layers.Conv2D(512, (2, 2), padding='valid', kernel_initializer=initializer, name='conv8_2')(up6_1) # 6 - 2
    conv6_3 = Norm()(up6_2)
    conv6_4 = tf.nn.relu(conv6_3)

    conv7_1 = tf.keras.layers.Conv2DTranspose(512, (4, 4), strides=(4, 2), padding='same', kernel_initializer=initializer, use_bias=False)(conv6_4)  # 24, 4
    crop_7 = tf.keras.layers.Cropping2D(cropping=((4, 4), (14, 14)), name='cropped_conv4_2')(conv4_1) # 32 - 24
    up7 = tf.keras.layers.concatenate([conv7_1, crop_7], axis=concat_axis, name='skip_connection2')
    up7_1 = tf.keras.layers.Conv2D(512, (3, 2), padding='valid', kernel_initializer=initializer, name='conv9_1')(up7)
    up7_2 = tf.keras.layers.Conv2D(512, (3, 2), padding='valid', kernel_initializer=initializer, name='conv9_2')(up7_1) # 20 2
    conv7_3 = Norm()(up7_2)
    conv7_4 = tf.nn.relu(conv7_3)

    conv8_1 = tf.keras.layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer, use_bias=False)(conv7_4) # 40, 4
    crop_8 = tf.keras.layers.Cropping2D(cropping=((12, 12), (30, 30)), name='cropped_conv3_2')(conv3_1) # 64 - 40
    up8 = tf.keras.layers.concatenate([conv8_1, crop_8], axis=concat_axis, name='skip_connection3')
    up8_1 = tf.keras.layers.Conv2D(256, (3, 2), padding='valid', kernel_initializer=initializer, name='conv10_1')(up8) # 38 3
    up8_2 = tf.keras.layers.Conv2D(256, (3, 2), padding='valid', kernel_initializer=initializer, name='conv10_2')(up8_1) # 36 2
    conv8_3 = Norm()(up8_2)
    conv8_4 = tf.nn.relu(conv8_3)

    conv9_1 = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer, use_bias=False)(conv8_4)  # 72, 4
    crop_9 = tf.keras.layers.Cropping2D(cropping=((28, 28), (62, 62)), name='cropped_conv2_2')(conv2_1)  # 128 - 72
    up9 = tf.keras.layers.concatenate([conv9_1, crop_9], axis=concat_axis, name='skip_connection4')
    up9_1 = tf.keras.layers.Conv2D(128, (3, 1), padding='valid', kernel_initializer=initializer, name='conv11_1')(up9) # 70, 4
    up9_2 = tf.keras.layers.Conv2D(128, (3, 1), padding='valid', kernel_initializer=initializer, name='conv11_2')(up9_1) # 68 4
    conv9_3 = Norm()(up9_2)
    conv9_4 = tf.nn.relu(conv9_3)

    conv10_1 = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), kernel_initializer=initializer, padding='same', use_bias=False)(conv9_4)  # 136, 8
    crop_10 = tf.keras.layers.Cropping2D(cropping=((60, 60), (124, 124)), name='cropped_conv1_2')(conv1_1)  # 256 - 136
    up10 = tf.keras.layers.concatenate([conv10_1, crop_10], axis=concat_axis, name='skip_connection5')
    up10_1 = tf.keras.layers.Conv2D(64, (3, 2), padding='valid', kernel_initializer=initializer, name='conv12_1')(up10) # 134, 7
    up10_2 = tf.keras.layers.Conv2D(64, (3, 2), padding='valid', kernel_initializer=initializer, name='conv12_2')(up10_1) # 123, 6
    conv10_3 = Norm()(up10_2)
    conv10_4 = tf.nn.relu(conv10_3)

    conv11_1 = tf.keras.layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer, use_bias=False)(conv10_4)  # 264, 12
    crop_11 = tf.keras.layers.Cropping2D(cropping=((124, 124), (250, 250)), name='cropped_conv0_2')(conv0_1)  # 512 - 264
    up11 = tf.keras.layers.concatenate([conv11_1, crop_11], axis=concat_axis, name='skip_connection6')
    up11_1 = tf.keras.layers.Conv2D(32, (4, 2), padding='valid', name='conv13_1')(up11) # 261, 11
    conv11_3 = Norm()(up11_1)
    conv11_4 = tf.nn.relu(conv11_3)

    h = tf.pad(conv11_4, [[0, 0], [1, 1], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(1, 7, kernel_initializer=initializer, padding='valid')(h)
    h = tf.nn.relu(h)

    model = tf.keras.Model(inputs = [inputs], outputs = [h])
    model.summary()

    return model

generator = Generator()
LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#Generator loss
def generator_loss(disc_generated_output, generated_image, target):

    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - generated_image))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss
#---------------------------------------------------------------------------------------

# build discriminator
def Discriminator():
    norm = 'batch_norm'
    Norm = _get_norm_layer(norm)
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02, seed=None)

    inp = keras.Input((1024, 1024, 3))
    inp_1 = tf.keras.layers.Cropping2D(cropping=((383, 384), (506, 507)), name='cropped_input')(inp)
    tar = keras.Input((257, 11, 1))
    inputs = tf.keras.layers.concatenate([inp_1, tar], axis=3, name='concatenation')

    h = keras.layers.Conv2D(64, 4, strides=(2, 2), kernel_initializer=initializer, padding='same')(inputs)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    h = keras.layers.Conv2D(128, 4, strides=(2, 2), kernel_initializer=initializer, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    h = keras.layers.Conv2D(256, 4, strides=(2, 1), kernel_initializer=initializer, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)


    zero_pad1 = tf.keras.layers.ZeroPadding2D()(h)  # (35, 5, 256)
    h = tf.keras.layers.Conv2D(512, (5, 4), strides=1,
                               kernel_initializer=initializer,
                               use_bias=False)(zero_pad1)  # (31, 2, 512)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)


    zero_pad2 = tf.keras.layers.ZeroPadding2D()(h)  # (33, 4, 512)
    h = tf.keras.layers.Conv2D(1, 4, strides=1,
                               kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)


    model = tf.keras.Model(inputs = [inp, tar], outputs=[h])

    model.summary()

    return model

discriminator = Discriminator()

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output) # real data tensor -> 1

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output) # fake data tensor -> 0

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss
#---------------------------------------------------------------------------------------
critic = 5
clip_value = 0.01

# Optimizer
generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)

checkpoint_dir = './training_checkpoints/FrictGAN'
checkpoint_dir1 = './saved_model'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                 discriminator_optimizer = discriminator_optimizer,
                                 generator = generator,
                                 discriminator = discriminator)

def generate_output(model, test_input):

  prediction = model(test_input, training=True)

  return prediction

def generate_images(model, test_input, tar):  #'model' is the generator
  prediction = model(test_input, training=True) # generate the output image
  pd = tf.reshape(prediction, [257, 11])

  plt.figure()
  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  print(prediction)
  pd = tf.reshape(prediction, [257, 11])
  print(tar)
  tar = tf.reshape(tar, [257, 11])
  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    #print(display_list[i])
    if i == 0:
        plt.imshow(display_list[i] * 0.5 + 0.5)
        print('0')
    else:
        display_show = tf.reshape(display_list[i], [257,11])
        plt.imshow(display_show * 0.5 + 0.5)
        #plt.imshow(display_list[i].numpy().squeeze() * 0.5 + 0.5, cmap=plt.cm.gray_r)
        print('B')
    plt.axis('off')
    np.savetxt("pre_spec.txt", pd)
    np.savetxt("tar.txt", tar)
  plt.show()

  S = np.loadtxt('./pre_spec.txt')
  T = np.loadtxt('./tar.txt')
  y_inv = np.abs(librosa.core.griffinlim(S))
  y_inv1 = np.abs(librosa.core.griffinlim(T))
  print('.............................................................')
  print(y_inv)
  print('.............................................................')
  print(y_inv1)
  print('.............................................................')
  err = np.sum((y_inv - y_inv1) ** 2)
  print(err/1280.0)

  plt.figure()
  plt.subplot(2, 1, 1)
  # librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time') #log for frequency
  librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max))  # No log for frequency
  plt.title('Generated Signals')
  plt.colorbar(format='%+2.0f dB')
  plt.tight_layout()

  plt.subplot(2, 1, 2)
  # librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time') #log for frequency
  librosa.display.specshow(librosa.amplitude_to_db(T, ref=np.max))  # No log for frequency
  plt.title('Original Signals')
  plt.colorbar(format='%+2.0f dB')
  plt.tight_layout()
  plt.show()

  n = 1280
  x = range(0,n)
  b, a = signal.butter(8, 0.02, 'lowpass')
  filte_y_inv = signal.filtfilt(b, a, y_inv)
  filte_y_inv1 = signal.filtfilt(b, a, y_inv1)

  plt.subplot(4, 1, 1)
  ax = plt.gca()
  ax.plot(x[:1280], y_inv[:1280])
  #ax.set_xlabel('Displacement')
  plt.title('Generated Signals')

  plt.subplot(4, 1, 2)
  ax = plt.gca()
  ax.plot(x[:1280], y_inv1[:1280])
  #ax.set_xlabel('Displacement')
  plt.title('Original Signals')

  plt.subplot(4, 1, 3)
  ax = plt.gca()
  ax.plot(x[:1280], filte_y_inv[:1280])

  plt.subplot(4, 1, 4)
  ax = plt.gca()
  ax.plot(x[:1280], filte_y_inv1[:1280])
  plt.show()



@tf.function
#def train_step_gen(input_image, target, test_show, train_show, epoch):
def train_step_gen(input_image, target, epoch):
  with tf.GradientTape() as gen_tape:

      gen_output = generator(input_image, training=True)  # initial spectrogram

      disc_generated_output = discriminator([input_image, gen_output], training=True)  # When Input fake, output fake score

      gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)

  generator_gradients = gen_tape.gradient(target=gen_total_loss, sources=generator.trainable_variables)
  generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))


  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    #tf.summary.image('valid image', ((sample_trainimage * 0.5) + 0.5), step=epoch, max_outputs=3)
    #tf.summary.image('fake valid spec', ((train_show * 0.5) + 0.5), step=epoch, max_outputs=3)
    #tf.summary.image('real valid spec', ((sample_trainspec * 0.5) + 0.5), step=epoch, max_outputs=3)
    #tf.summary.image('real test spec', ((sample_spec * 0.5) + 0.5), step=epoch, max_outputs=3)
    #tf.summary.image('fake test spec', ((test_show * 0.5) + 0.5), step=epoch, max_outputs=3)
    #tf.summary.image('test image', ((sample_image * 0.5) + 0.5), step=epoch, max_outputs=3)

  return gen_total_loss, gen_gan_loss, gen_l1_loss

# discriminator train
def train_step_dis(input_image, target, epoch):
  with tf.GradientTape() as disc_tape:

      gen_output = generator(input_image, training=True)

      disc_real_output = discriminator([input_image, target], training=True)

      disc_generated_output = discriminator([input_image, gen_output], training=True)

      disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  discriminator_gradients = disc_tape.gradient(target=disc_loss, sources=discriminator.trainable_variables)
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

  for w in discriminator.trainable_variables:
      w.assign(tf.clip_by_value(w, -clip_value, clip_value))

  with summary_writer.as_default():
      tf.summary.scalar('disc_loss', disc_loss, step=epoch)

  return disc_loss

# training process
def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()

      # GENERATE DATA
      #gt_image = generate_output(generator, sample_trainimage)
      #g_image = generate_output(generator, sample_image)

      for i in range(critic):
         D = train_step_dis(input_image, target, epoch)
      # TL, G, L1 = train_step_gen(input_image, target, gt_image, g_image, epoch)  # train generator
      TL, G, L1 = train_step_gen(input_image, target, epoch)  # train generator

      print("D_loss: {:.2f}".format(D), "G_loss: {:.2f}".format(G), "gen_l1_loss {:.2f}".format(L1), "gen_total_loss: {:.2f}".format(TL))

    if (epoch + 1) % 50 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
#fit(train_dataset, EPOCHS, test_dataset)

for inp, tar in valid_dataset.take(24):
   generate_images(generator, inp, tar)


