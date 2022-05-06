from keras.models import Model
from keras.layers.core import Activation
from keras.layers import Input, add, multiply, Lambda, Dense
from keras.layers.convolutional import Conv2D
from keras import backend as K

import tensorflow as tf
from collections import defaultdict
from tqdm import tqdm

def gelu(x):
    cdf = 0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0)))
    return x * cdf

def fft2d(input, gamma=0.1):
    temp = K.permute_dimensions(input, (0, 3, 1, 2))
    fft = tf.fft2d(tf.complex(temp, tf.zeros_like(temp)))
    absfft = tf.pow(tf.abs(fft)+1e-8, gamma)
    output = K.permute_dimensions(absfft, (0, 2, 3, 1))
    return output

def fftshift2d(input, size_psc=128):
    bs, h, w, ch = input.get_shape().as_list()
    fs11 = input[:, -h // 2:h, -w // 2:w, :]
    fs12 = input[:, -h // 2:h, 0:w // 2, :]
    fs21 = input[:, 0:h // 2, -w // 2:w, :]
    fs22 = input[:, 0:h // 2, 0:w // 2, :]
    output = tf.concat([tf.concat([fs11, fs21], axis=1), tf.concat([fs12, fs22], axis=1)], axis=2)
    output = tf.image.resize_images(output, (size_psc, size_psc), 0)
    return output

def pixel_shiffle(layer_in, scale):
    return tf.depth_to_space(layer_in, block_size=scale)

def global_average_pooling2d(layer_in):
    return tf.reduce_mean(layer_in, axis=(1, 2), keepdims=True)

def gen_dot_dic():
    dot_dic = {}
    for k in range(64):
        k_1, k_2 = k, 127 - k
        for i in range(0, k_2 - k_1 + 1):
            dot_dic[(k_1, k_1 + i)] = k
            dot_dic[(k_1 + i, k_1)] = k
            dot_dic[(k_2 - i, k_2)] = k
            dot_dic[(k_2, k_2 - i)] = k
    return dot_dic

def Void_Conv(input, channel):
    for i in range(64):
        slice_1 = tf.concat([input[:, i: i + 1, i: 128 - i, :],
                           input[:, 127 - i: 128 - i, i: 128 - i, :]], axis=2)
        slice_2 = tf.transpose(tf.concat([input[:, i: 128 - i, i: i + 1, :],
                                          input[:, i: 128 - i, 127 - i: 128 - i, :]], axis=1), perm=[0, 2, 1, 3])
        slice = tf.squeeze(tf.reduce_mean(tf.concat([slice_1, slice_2], axis=2), axis=3), axis=1)
        shape = int(slice.shape[1])
        slice_out = Dense(shape // 32, input_shape=(shape, ))(slice)

        if i == 0:
            input_layer = slice_out
        else:
            input_layer = tf.concat([input_layer, slice_out], axis=1)

    return input_layer


def re(input):
    return tf.reshape(input, [-1, 128, 128, 1])

def MF(input):
    dim = 10
    W1 = tf.Variable(tf.random_normal([int(input.shape[1]), dim]))
    W2 = tf.Variable(tf.random_normal([dim, 128 * 128]))
    bias = tf.Variable(tf.random_normal([1, 128 * 128]))
    output = tf.matmul(input, tf.matmul(W1, W2)) + bias
    output = tf.nn.relu(output)
    return output

def FCALayer(input, channel, reduction=16, size_psc=128, dot_dic=''):
    absfft1 = Lambda(fft2d, arguments={'gamma': 0.8})(input)
    absfft1 = Lambda(fftshift2d, arguments={'size_psc': size_psc})(absfft1)    # bs * 128 * 128 * 64

    dnn_input = Lambda(Void_Conv, arguments={'channel': 64})(absfft1)
    shape = int(dnn_input.shape[1])

    dnn_input = Dense(256, input_shape=(shape, ))(dnn_input)

    dnn_output = Lambda(MF)(dnn_input)
    # dnn_output = Dense(128 * 128, input_shape=(shape,))(dnn_input)

    absfft1 = Lambda(re)(dnn_output)
    print(absfft1.shape)

    absfft2 = Conv2D(channel, kernel_size=3, activation='relu', padding='same')(absfft1)
    W = Lambda(global_average_pooling2d)(absfft2)
    W = Conv2D(channel // reduction, kernel_size=1, activation='relu', padding='same')(W)
    W = Conv2D(channel, kernel_size=1, activation='sigmoid', padding='same')(W)
    mul = multiply([input, W])
    return mul


def FCAB(input, channel, size_psc=128, dot_dic=''):
    conv = Conv2D(channel, kernel_size=3, padding='same')(input)
    conv = Lambda(gelu)(conv)
    conv = Conv2D(channel, kernel_size=3, padding='same')(conv)
    conv = Lambda(gelu)(conv)
    att = FCALayer(conv, channel, reduction=16, size_psc=size_psc, dot_dic=dot_dic)
    output = add([att, input])
    return output


def ResidualGroup(input, channel, size_psc=128, dot_dic=''):
    conv = input
    n_RCAB = 3
    for _ in range(n_RCAB):
        print('Fourier block: {}'.format(_))
        conv = FCAB(conv, channel=channel, size_psc=size_psc, dot_dic=dot_dic)
    conv = add([conv, input])
    return conv


def EDCSNN(input_shape, scale=2, size_psc=128):
    dot_dic = gen_dot_dic()
    inputs = Input(input_shape)
    conv = Conv2D(64, kernel_size=3, padding='same')(inputs)
    conv = Lambda(gelu)(conv)
    n_ResGroup = 3
    for _ in range(n_ResGroup):
        print('ResidualGroup: {}'.format(_ + 1))
        conv = ResidualGroup(conv, 64, size_psc, dot_dic)
    conv = Conv2D(64 * (scale ** 2), kernel_size=3, padding='same')(conv)
    conv = Lambda(gelu)(conv)
    upsampled = Lambda(pixel_shiffle, arguments={'scale': scale})(conv)
    conv = Conv2D(1, kernel_size=3, padding='same')(upsampled)
    output = Activation('sigmoid')(conv)

    model = Model(inputs=inputs, outputs=output)
    return model
