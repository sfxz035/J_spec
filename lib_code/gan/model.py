import numpy as np
import scipy.io
import cv2
import scipy.misc
import scipy.io as sio
from scipy.signal import convolve2d
import tensorflow as tf
import time
import os
import argparse
import random
#tf.app.flags.DEFINE_string("model_file", "models/train/unet2_2_1024_339.ckpt-done", "")

def relu(input):
    relu = tf.nn.relu(input)
    # convert nan to zero (nan != nan)
    nan_to_zero = tf.where(tf.equal(relu, relu), relu, tf.zeros_like(relu))
    return nan_to_zero

def conv3d(x, input_filters, output_filters, kernel, strides,name):
    with tf.variable_scope(name):
        shape = [kernel,kernel, kernel, input_filters, output_filters]
        weight = tf.get_variable('w', shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),regularizer=None, trainable=True, collections=None)
        return tf.nn.conv3d(x, weight, strides=[1, strides,strides, strides, 1], padding='SAME',name='conv')
def conv3d_b(x, input_filters, output_filters, kernel, strides,name ):
    with tf.variable_scope(name):
        shape = [kernel, kernel, kernel, input_filters, output_filters]
        bias_shape = [output_filters]
        weight = tf.get_variable('w', shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),regularizer=None, trainable=True, collections=None)
        biases = tf.get_variable('b', shape=bias_shape, dtype=tf.float32, initializer=tf.zeros_initializer(),regularizer=None, trainable=True, collections=None)
        return tf.nn.conv3d(x, weight, strides=[1, strides, strides, strides, 1], padding='SAME',
                            name='conv')+biases
def deconv3d(x, input_filters, output_filters, kernel, strides,name):
    with tf.variable_scope(name):
        shape = [kernel, kernel,kernel, output_filters,input_filters]
        weight = tf.get_variable('w', shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), regularizer=None, trainable=True, collections=None)
        batch_size = tf.shape(x)[0]
        depth=tf.shape(x)[1] * strides
        height = tf.shape(x)[2] * strides
        width = tf.shape(x)[3] * strides
        outshape=[batch_size,depth,height,width,output_filters]
        return tf.nn.conv3d_transpose(x, weight, output_shape=outshape,strides=[1, strides,strides, strides, 1], padding='SAME',name='deconv')
def deconv3d_b(x, input_filters, output_filters, kernel, strides,name):
    with tf.variable_scope(name):
        shape = [kernel, kernel,kernel, output_filters,input_filters]
        bias_shape = [output_filters]
        weight = tf.get_variable('w', shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), regularizer=None, trainable=True, collections=None)
        biases = tf.get_variable('b', shape=bias_shape, dtype=tf.float32, initializer=tf.zeros_initializer(), regularizer=None, trainable=True, collections=None)
        batch_size = tf.shape(x)[0]
        depth=tf.shape(x)[1] * strides
        height = tf.shape(x)[2] * strides
        width = tf.shape(x)[3] * strides
        outshape=[batch_size,depth,height,width,output_filters]
        return tf.nn.conv3d_transpose(x, weight, output_shape=outshape,strides=[1, strides,strides, strides, 1], padding='SAME',
                            name='deconv')+biases
def max_pool3d(input):
  return tf.nn.max_pool3d(input, ksize=[1,2, 2, 2, 1], strides=[1,2, 2, 2, 1], padding='SAME')

def unet3d(input):
    with tf.variable_scope('unet3d'):
        encode1_1=relu(tf.layers.batch_normalization(conv3d(input,1,32,5,1,'encode1_1'), training=True, name='bnen1'))
        encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=True, name='bnen1_2'))
        encode1_3 = relu(tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 1, 'encode1_3'), training=True, name='bnen1_3')+encode1_1)

        encode2_1=max_pool3d(encode1_3)
        encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=True, name='bnen2_2'))
        encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=True, name='bnen2_3'))
        encode2_4 = relu(tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 1, 'encode2_4'), training=True, name='bnen2_4')+encode2_2)

        encode3_1=max_pool3d(encode2_4)
        encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=True, name='bnen3_2'))
        encode3_3 = relu(tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=True, name='bnen3_3'))
        encode3_4 = relu(tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 1, 'encode3_4'), training=True, name='bnen3_4')+encode3_2)

        encode4_1 = max_pool3d(encode3_4)
        encode4_2 = relu(tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_2'), training=True, name='bnen4_2'))
        encode4_3 = relu(tf.layers.batch_normalization(conv3d(encode4_2, 256, 256, 3, 1, 'encode4_3'), training=True, name='bnen4_3'))
        encode4_4 = relu(tf.layers.batch_normalization(conv3d(encode4_3, 256, 256, 3, 1, 'encode4_4'), training=True, name='bnen4_4')+encode4_2)

        decode3_1=relu(tf.layers.batch_normalization(deconv3d(encode4_4, 256, 128, 3, 2, 'decode3_1'), training=True, name='bnde3_1'))
        decode3_2 = tf.concat([decode3_1, encode3_4], 4)
        decode3_3 = relu(tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_3'), training=True, name='bnde3_3'))
        decode3_4 = relu(tf.layers.batch_normalization(conv3d(decode3_3, 128, 128, 3, 1, 'decode3_4'), training=True, name='bnde3_4'))
        decode3_5 = relu(tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_5'), training=True, name='bnde3_5')+decode3_3)

        decode2_1 = relu(tf.layers.batch_normalization(deconv3d(decode3_5, 128, 64, 3, 2, 'decode2_1'), training=True, name='bnde2_1'))
        decode2_2 = tf.concat([decode2_1, encode2_4], 4)
        decode2_3 = relu(tf.layers.batch_normalization(conv3d(decode2_2, 128, 64, 3, 1, 'decode2_3'), training=True, name='bnde2_3'))
        decode2_4 = relu(tf.layers.batch_normalization(conv3d(decode2_3, 64, 64, 3, 1, 'decode2_4'), training=True, name='bnde2_4'))
        decode2_5 = relu(tf.layers.batch_normalization(conv3d(decode2_4, 64, 64, 3, 1, 'decode2_5'), training=True, name='bnde2_5')+decode2_3)

        decode1_1 = relu(tf.layers.batch_normalization(deconv3d(decode2_5, 64, 32, 3, 2, 'decode1_1'), training=True, name='bnde1_1'))
        decode1_2 = tf.concat([decode1_1, encode1_3], 4)
        decode1_3 = relu(tf.layers.batch_normalization(conv3d(decode1_2, 64, 32, 3, 1, 'decode1_3'), training=True, name='bnde1_3'))
        decode1_4 = relu(tf.layers.batch_normalization(conv3d(decode1_3, 32, 32, 3, 1, 'decode1_4'), training=True, name='bnde1_4'))
        decode1_5 = relu(tf.layers.batch_normalization(conv3d(decode1_4, 32, 32, 3, 1, 'decode1_5'), training=True, name='bnde1_5')+decode1_3)
        output=conv3d_b(decode1_5, 32, 1, 3, 1, 'output')

        return output
def quweidu(x):
    x=x[0]
    x=x[:,:,:,0]
    return x
def conv2d_transpose(x, input_filters, output_filters, kernel, strides):
    with tf.variable_scope('conv_transpose'):

        shape = [kernel, kernel, output_filters, input_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1] * strides
        width = tf.shape(x)[2] * strides
        output_shape = tf.stack([batch_size, height, width, output_filters])
        return tf.nn.conv2d_transpose(x, weight, output_shape, strides=[1, strides, strides, 1], name='conv_transpose')
def Feature_Extraction(Im):
    # r = len(Im.get_shape()) - 2
    #
    # if r == 2:
    #     psf1 = [[0, 0, 0], [-1 / 3, 2 / 3, -1 / 3], [0, 0, 0]]
    #     psf1 = tf.reshape(psf1, [3, 3, 1, 1])
    #     psf2 = [[-1 / 3, 0, 0], [0, 2 / 3, 0], [0, 0, -1 / 3]]
    #     psf2 = tf.reshape(psf2, [3, 3, 1, 1])
    #     psf3 = [[0, -1 / 3, 0], [0, 2 / 3, 0], [0, -1 / 3, 0]]
    #     psf3 = tf.reshape(psf3, [3, 3, 1, 1])
    #     psf4 = [[0, 0, -1 / 3], [0, 2 / 3, 0], [-1 / 3, 0, 0]]
    #     psf4 = tf.reshape(psf4, [3, 3, 1, 1])
    #
    #     psfs = tf.concat([psf1, psf2, psf3, psf4], 3)

    # elif r == 3:
    # filt_siz = np.array([1, 1, 1]) * 3
    # sig = np.array([1, 1, 1]) * 0.6
    #
    # siz = (filt_siz - 1) / 2
    # (x, y, z) = np.mgrid[-siz[0]:siz[0] + 1, -siz[1]:siz[1] + 1, -siz[2]:siz[2] + 1]
    #
    # psf0 = np.exp(-(np.square(x) / 2 / np.square(sig[0]) + np.square(y) / 2 / np.square(sig[1]) + np.square(
    #     z) / 2 / np.square(sig[2])))
    # psf0 = psf0 / np.sum(psf0)
    # psf0 = tf.reshape(psf0, [3, 3, 3, 1, 1])
    # psf0 = tf.cast(psf0, tf.float32)

    psf1 = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [-1 / 3, 2 / 3, -1 / 3], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    psf1 = tf.reshape(psf1, [3, 3, 3, 1, 1])
    psf1 = tf.cast(psf1, tf.float32)
    psf2 = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, -1 / 3, 0], [0, 2 / 3, 0], [0, -1 / 3, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    psf2 = tf.reshape(psf2, [3, 3, 3, 1, 1])
    psf2 = tf.cast(psf2, tf.float32)
    psf3 = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[-1 / 3, 0, 0], [0, 2 / 3, 0], [0, 0, -1 / 3]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    psf3 = tf.reshape(psf3, [3, 3, 3, 1, 1])
    psf3 = tf.cast(psf3, tf.float32)
    psf4 = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, -1 / 3], [0, 2 / 3, 0], [-1 / 3, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    psf4 = tf.reshape(psf4, [3, 3, 3, 1, 1])
    psf4 = tf.cast(psf4, tf.float32)

    psf5 = np.array([[[0, 0, 0], [0, -1 / 3, 0], [0, 0, 0]], [[0, 0, 0], [0, 2 / 3, 0], [0, 0, 0]],
                     [[0, 0, 0], [0, -1 / 3, 0], [0, 0, 0]]])
    psf5 = tf.reshape(psf5, [3, 3, 3, 1, 1])
    psf5 = tf.cast(psf5, tf.float32)
    psf6 = np.array([[[0, 0, 0], [-1 / 3, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 2 / 3, 0], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, -1 / 3], [0, 0, 0]]])
    psf6 = tf.reshape(psf6, [3, 3, 3, 1, 1])
    psf6 = tf.cast(psf6, tf.float32)
    psf7 = np.array([[[0, 0, 0], [0, 0, -1 / 3], [0, 0, 0]], [[0, 0, 0], [0, 2 / 3, 0], [0, 0, 0]],
                     [[0, 0, 0], [-1 / 3, 0, 0], [0, 0, 0]]])
    psf7 = tf.reshape(psf7, [3, 3, 3, 1, 1])
    psf7 = tf.cast(psf7, tf.float32)

    psf8 = np.array([[[0, -1 / 3, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 2 / 3, 0], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, -1 / 3, 0]]])
    psf8 = tf.reshape(psf8, [3, 3, 3, 1, 1])
    psf8 = tf.cast(psf8, tf.float32)
    psf9 = np.array([[[0, 0, 0], [0, 0, 0], [0, -1 / 3, 0]], [[0, 0, 0], [0, 2 / 3, 0], [0, 0, 0]],
                     [[0, -1 / 3, 0], [0, 0, 0], [0, 0, 0]]])
    psf9 = tf.reshape(psf9, [3, 3, 3, 1, 1])
    psf9 = tf.cast(psf9, tf.float32)

    psfs = tf.concat([ psf1, psf2, psf3, psf4, psf5, psf6, psf7, psf8, psf9], 4)

    ImFea = tf.nn.convolution(Im, psfs, padding='SAME', strides=None, dilation_rate=None)

    return ImFea
def synthesis_ref(maginput,fieldinput,tkdinput):
    with tf.variable_scope('syn'):
        #input2=tf.concat([maginput,fieldinput,tkdinput],4)
        # input2_112= tf.reshape(input2,[32,32,32])
        # input2_56 = tf.image.resize_images(input2, [16, 16,16])
        # input2_28 = tf.image.resize_images(input2, [8, 8,8])
        magconv1=relu(tf.layers.batch_normalization(conv3d(maginput,1,16,5,1,'magconv1'), training=True, name='bnmag1'))
        magconv2_1 = relu(tf.layers.batch_normalization(conv3d(magconv1, 16, 16, 3, 1, 'magconv2_1'), training=True, name='bnmag2_1'))
        magconv2_2 = relu(tf.layers.batch_normalization(conv3d(magconv2_1, 16, 16, 3, 1, 'magconv2_2')+magconv1, training=True, name='bnmag2_2'))
        magconv3_1 = relu(tf.layers.batch_normalization(conv3d(magconv2_2, 16, 16, 3, 1, 'magconv3_1'), training=True, name='bnmag3_1'))
        magconv3_2 = relu(tf.layers.batch_normalization(conv3d(magconv3_1, 16, 16, 3, 1, 'magconv3_2') + magconv2_2, training=True, name='bnmag3_2'))
        magconv4_1 = relu(tf.layers.batch_normalization(conv3d(magconv3_2, 16, 16, 3, 1, 'magconv4_1'), training=True, name='bnmag4_1'))
        magconv4_2 = relu(tf.layers.batch_normalization(conv3d(magconv4_1, 16, 16, 3, 1, 'magconv4_2') + magconv3_2, training=True, name='bnmag4_2'))
        magconv5_1 = relu(tf.layers.batch_normalization(conv3d(magconv4_2, 16, 16, 3, 1, 'magconv5_1'), training=True, name='bnmag5_1'))
        magconv5_2 = relu(tf.layers.batch_normalization(conv3d(magconv5_1, 16, 16, 3, 1, 'magconv5_2') + magconv4_2, training=True, name='bnmag5_2'))

        tkdconv1 = relu(
            tf.layers.batch_normalization(conv3d(tkdinput, 1, 16, 5, 1, 'tkdconv1'), training=True, name='bntkd1'))
        tkdconv2_1 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv1, 16, 16, 3, 1, 'tkdconv2_1'), training=True, name='bntkd2_1'))
        tkdconv2_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv2_1, 16, 16, 3, 1, 'tkdconv2_2') + tkdconv1, training=True,
                                          name='bntkd2_2'))
        tkdconv3_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv2_2, 16, 16, 3, 1, 'tkdconv3_1'), training=True,
                                                        name='bntkd3_1'))
        tkdconv3_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv3_1, 16, 16, 3, 1, 'tkdconv3_2') + tkdconv2_2, training=True,
                                          name='bntkd3_2'))
        tkdconv4_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv3_2, 16, 16, 3, 1, 'tkdconv4_1'), training=True,
                                                        name='bntkd4_1'))
        tkdconv4_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv4_1, 16, 16, 3, 1, 'tkdconv4_2') + tkdconv3_2, training=True,
                                          name='bntkd4_2'))
        tkdconv5_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv4_2, 16, 16, 3, 1, 'tkdonv5_1'), training=True,
                                                        name='bntkd5_1'))
        tkdconv5_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv5_1, 16, 16, 3, 1, 'tkdconv5_2') + tkdconv4_2, training=True,
                                          name='bntkd5_2'))

        fieldconv1=relu(tf.layers.batch_normalization(conv3d(fieldinput,1,16,5,1,'fieldconv1'), training=True, name='bnfield1'))
        fieldconv2_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv1, 16, 16, 3, 1, 'fieldconv2_1'), training=True, name='bndield2_1'))
        fieldconv2_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_1, 16, 16, 3, 1, 'fieldconv2_2')+fieldconv1, training=True, name='bnfield2_2'))
        fieldconv3_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_2, 16, 16, 3, 1, 'fieldconv3_1'), training=True, name='bnfield3_1'))
        fieldconv3_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_1, 16, 16, 3, 1, 'fieldconv3_2') + fieldconv2_2, training=True, name='bnfield3_2'))
        fieldconv4_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_2, 16, 16, 3, 1, 'fieldconv4_1'), training=True, name='bnfield4_1'))
        fieldconv4_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_1, 16, 16, 3, 1, 'fieldconv4_2') + fieldconv3_2, training=True, name='bnfield4_2'))
        fieldconv5_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_2, 16, 16, 3, 1, 'fieldconv5_1'), training=True, name='bnfield5_1'))
        fieldconv5_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv5_1, 16, 16, 3, 1, 'fieldconv5_2') + fieldconv4_2, training=True, name='bnfield5_2'))
        encodeinput=tf.concat([magconv5_2,fieldconv5_2,tkdconv5_2],4)
        encode1_1=relu(tf.layers.batch_normalization(conv3d(encodeinput,48,32,3,1,'encode1_1'), training=True, name='bnen1'))
        encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=True, name='bnen1_2'))
        encode1_3 = relu(tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 1, 'encode1_3'), training=True, name='bnen1_3'))

        encode2_1=max_pool3d(encode1_3)
        encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=True, name='bnen2_2'))
        encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=True, name='bnen2_3'))
        encode2_4 = relu(tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 1, 'encode2_4'), training=True, name='bnen2_4'))

        encode3_1=max_pool3d(encode2_4)
        encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=True, name='bnen3_2'))
        encode3_3 = relu(tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=True, name='bnen3_3'))
        encode3_4 = relu(tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 1, 'encode3_4'), training=True, name='bnen3_4'))

        encode4_1 = max_pool3d(encode3_4)
        encode4_3 = relu(tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_3'), training=True, name='bnen4_3'))
        encode4_4 = relu(tf.layers.batch_normalization(conv3d(encode4_3, 256, 256, 3, 1, 'encode4_4'), training=True, name='bnen4_4'))
        encode4_5 = relu(tf.layers.batch_normalization(conv3d(encode4_4, 256, 256, 3, 1, 'encode4_5'), training=True, name='bnen4_5'))

        decode3_1 = relu(tf.layers.batch_normalization(deconv3d(encode4_5, 256, 128, 3, 2, 'decode3_1'), training=True,
                                                       name='bnde3_1'))
        decode3_2=tf.concat([decode3_1,encode3_4],4)
        decode3_4 = relu(tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=True, name='bnde3_4'))
        decode3_5 = relu(tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_5'), training=True, name='bnde3_5'))
        decode3_6 = relu(tf.layers.batch_normalization(conv3d(decode3_5, 128, 128, 3, 1, 'decode3_6'), training=True, name='bnde3_6'))

        decode4_1 = relu(tf.layers.batch_normalization(deconv3d(decode3_6, 128, 64, 3, 2, 'decode2_1'), training=True,
                                          name='bnde2_1'))
        decode4_2 = tf.concat([decode4_1, encode2_4], 4)
        decode4_4 = relu(tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=True, name='bnde4_4'))
        decode4_5 = relu(tf.layers.batch_normalization(conv3d(decode4_4, 64, 64, 3, 1, 'decode4_5'), training=True, name='bnde4_5'))
        decode4_6 = relu(tf.layers.batch_normalization(conv3d(decode4_5, 64, 64, 3, 1, 'decode4_6'), training=True, name='bnde4_6'))

        decode5_1 = relu(tf.layers.batch_normalization(deconv3d(decode4_6, 64, 32, 3, 2, 'decode1_1'), training=True,
                                                       name='bnde1_1'))
        decode5_2 = tf.concat([decode5_1, encode1_3], 4)
        decode5_4 = relu(tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=True, name='bnde5_4'))
        decode5_5 = relu(tf.layers.batch_normalization(conv3d(decode5_4, 32, 32, 3, 1, 'decode5_5'), training=True, name='bnde5_5'))
        decode5_6 = relu(tf.layers.batch_normalization(conv3d(decode5_5, 32, 32, 3, 1, 'decode5_6'), training=True, name='bnde5_6'))
        cosout=conv3d_b(decode5_6, 32, 1, 3, 1, 'out')+tkdinput


        return cosout
def synthesis_ref2(maginput,fieldinput,tkdinput):
    with tf.variable_scope('syn'):
        #input2=tf.concat([maginput,fieldinput,tkdinput],4)
        # input2_112= tf.reshape(input2,[32,32,32])
        # input2_56 = tf.image.resize_images(input2, [16, 16,16])
        # input2_28 = tf.image.resize_images(input2, [8, 8,8])
        magconv1=relu(tf.layers.batch_normalization(conv3d(maginput,1,16,5,1,'magconv1'), training=True, name='bnmag1'))
        magconv2_1 = relu(tf.layers.batch_normalization(conv3d(magconv1, 16, 16, 3, 1, 'magconv2_1'), training=True, name='bnmag2_1'))
        magconv2_2 = relu(tf.layers.batch_normalization(conv3d(magconv2_1, 16, 16, 3, 1, 'magconv2_2')+magconv1, training=True, name='bnmag2_2'))
        magconv3_1 = relu(tf.layers.batch_normalization(conv3d(magconv2_2, 16, 16, 3, 1, 'magconv3_1'), training=True, name='bnmag3_1'))
        magconv3_2 = relu(tf.layers.batch_normalization(conv3d(magconv3_1, 16, 16, 3, 1, 'magconv3_2') + magconv2_2, training=True, name='bnmag3_2'))
        magconv4_1 = relu(tf.layers.batch_normalization(conv3d(magconv3_2, 16, 16, 3, 1, 'magconv4_1'), training=True, name='bnmag4_1'))
        magconv4_2 = relu(tf.layers.batch_normalization(conv3d(magconv4_1, 16, 16, 3, 1, 'magconv4_2') + magconv3_2, training=True, name='bnmag4_2'))
        magconv5_1 = relu(tf.layers.batch_normalization(conv3d(magconv4_2, 16, 16, 3, 1, 'magconv5_1'), training=True, name='bnmag5_1'))
        magconv5_2 = relu(tf.layers.batch_normalization(conv3d(magconv5_1, 16, 16, 3, 1, 'magconv5_2') + magconv4_2, training=True, name='bnmag5_2'))

        tkdconv1 = relu(
            tf.layers.batch_normalization(conv3d(tkdinput, 1, 16, 5, 1, 'tkdconv1'), training=True, name='bntkd1'))
        tkdconv2_1 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv1, 16, 16, 3, 1, 'tkdconv2_1'), training=True, name='bntkd2_1'))
        tkdconv2_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv2_1, 16, 16, 3, 1, 'tkdconv2_2') + tkdconv1, training=True,
                                          name='bntkd2_2'))
        tkdconv3_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv2_2, 16, 16, 3, 1, 'tkdconv3_1'), training=True,
                                                        name='bntkd3_1'))
        tkdconv3_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv3_1, 16, 16, 3, 1, 'tkdconv3_2') + tkdconv2_2, training=True,
                                          name='bntkd3_2'))
        tkdconv4_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv3_2, 16, 16, 3, 1, 'tkdconv4_1'), training=True,
                                                        name='bntkd4_1'))
        tkdconv4_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv4_1, 16, 16, 3, 1, 'tkdconv4_2') + tkdconv3_2, training=True,
                                          name='bntkd4_2'))
        tkdconv5_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv4_2, 16, 16, 3, 1, 'tkdonv5_1'), training=True,
                                                        name='bntkd5_1'))
        tkdconv5_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv5_1, 16, 16, 3, 1, 'tkdconv5_2') + tkdconv4_2, training=True,
                                          name='bntkd5_2'))

        fieldconv1=relu(tf.layers.batch_normalization(conv3d(fieldinput,1,16,5,1,'fieldconv1'), training=True, name='bnfield1'))
        fieldconv2_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv1, 16, 16, 3, 1, 'fieldconv2_1'), training=True, name='bndield2_1'))
        fieldconv2_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_1, 16, 16, 3, 1, 'fieldconv2_2')+fieldconv1, training=True, name='bnfield2_2'))
        fieldconv3_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_2, 16, 16, 3, 1, 'fieldconv3_1'), training=True, name='bnfield3_1'))
        fieldconv3_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_1, 16, 16, 3, 1, 'fieldconv3_2') + fieldconv2_2, training=True, name='bnfield3_2'))
        fieldconv4_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_2, 16, 16, 3, 1, 'fieldconv4_1'), training=True, name='bnfield4_1'))
        fieldconv4_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_1, 16, 16, 3, 1, 'fieldconv4_2') + fieldconv3_2, training=True, name='bnfield4_2'))
        fieldconv5_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_2, 16, 16, 3, 1, 'fieldconv5_1'), training=True, name='bnfield5_1'))
        fieldconv5_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv5_1, 16, 16, 3, 1, 'fieldconv5_2') + fieldconv4_2, training=True, name='bnfield5_2'))
        encodeinput=tf.concat([magconv5_2,fieldconv5_2,tkdconv5_2],4)
        encode1_1=relu(tf.layers.batch_normalization(conv3d(encodeinput,48,32,3,1,'encode1_1'), training=True, name='bnen1'))
        encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=True, name='bnen1_2'))
        encode1_3 = relu(tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 1, 'encode1_3'), training=True, name='bnen1_3'))

        encode2_1=max_pool3d(encode1_3)
        encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=True, name='bnen2_2'))
        encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=True, name='bnen2_3'))
        encode2_4 = relu(tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 1, 'encode2_4'), training=True, name='bnen2_4'))

        encode3_1=max_pool3d(encode2_4)
        encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=True, name='bnen3_2'))
        encode3_3 = relu(tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=True, name='bnen3_3'))
        encode3_4 = relu(tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 1, 'encode3_4'), training=True, name='bnen3_4'))

        encode4_1 = max_pool3d(encode3_4)
        encode4_3 = relu(tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_3'), training=True, name='bnen4_3'))
        encode4_4 = relu(tf.layers.batch_normalization(conv3d(encode4_3, 256, 256, 3, 1, 'encode4_4'), training=True, name='bnen4_4'))
        encode4_5 = relu(tf.layers.batch_normalization(conv3d(encode4_4, 256, 256, 3, 1, 'encode4_5'), training=True, name='bnen4_5'))

        decode3_1 = relu(tf.layers.batch_normalization(deconv3d(encode4_5, 256, 128, 3, 2, 'decode3_1'), training=True,
                                                       name='bnde3_1'))
        decode3_2=tf.concat([decode3_1,encode3_4],4)
        decode3_4 = relu(tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=True, name='bnde3_4'))
        decode3_5 = relu(tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_5'), training=True, name='bnde3_5'))
        decode3_6 = relu(tf.layers.batch_normalization(conv3d(decode3_5, 128, 128, 3, 1, 'decode3_6'), training=True, name='bnde3_6'))

        decode4_1 = relu(tf.layers.batch_normalization(deconv3d(decode3_6, 128, 64, 3, 2, 'decode2_1'), training=True,
                                          name='bnde2_1'))
        decode4_2 = tf.concat([decode4_1, encode2_4], 4)
        decode4_4 = relu(tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=True, name='bnde4_4'))
        decode4_5 = relu(tf.layers.batch_normalization(conv3d(decode4_4, 64, 64, 3, 1, 'decode4_5'), training=True, name='bnde4_5'))
        decode4_6 = relu(tf.layers.batch_normalization(conv3d(decode4_5, 64, 64, 3, 1, 'decode4_6'), training=True, name='bnde4_6'))

        decode5_1 = relu(tf.layers.batch_normalization(deconv3d(decode4_6, 64, 32, 3, 2, 'decode1_1'), training=True,
                                                       name='bnde1_1'))
        decode5_2 = tf.concat([decode5_1, encode1_3], 4)
        decode5_4 = relu(tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=True, name='bnde5_4'))
        decode5_5 = relu(tf.layers.batch_normalization(conv3d(decode5_4, 32, 32, 3, 1, 'decode5_5'), training=True, name='bnde5_5'))
        decode5_6 = relu(tf.layers.batch_normalization(conv3d(decode5_5, 32, 32, 3, 1, 'decode5_6'), training=True, name='bnde5_6'))
        cosout=conv3d_b(decode5_6, 32, 1, 3, 1, 'out')+tkdinput


    with tf.variable_scope('ref'):
        fea = Feature_Extraction(cosout)
        refencodeinput = tf.concat([cosout, fea], 4)
        refencode1_1 = relu(
            tf.layers.batch_normalization(conv3d(refencodeinput, 10, 32, 3, 1, 'encode1_1'), training=True,
                                          name='bnen1'))
        refencode1_2 = relu(
            tf.layers.batch_normalization(conv3d(refencode1_1, 32, 32, 3, 1, 'encode1_2'), training=True,
                                          name='bnen1_2'))
        refencode1_3 = relu(
            tf.layers.batch_normalization(conv3d(refencode1_2, 32, 32, 3, 1, 'encode1_3') + refencode1_1, training=True,
                                          name='bnen1_3'))

        refencode2_1 = max_pool3d(refencode1_3)
        refencode2_2 = relu(
            tf.layers.batch_normalization(conv3d(refencode2_1, 32, 64, 3, 1, 'encode2_2'), training=True,
                                          name='bnen2_2'))
        refencode2_3 = relu(
            tf.layers.batch_normalization(conv3d(refencode2_2, 64, 64, 3, 1, 'encode2_3'), training=True,
                                          name='bnen2_3'))
        refencode2_4 = relu(
            tf.layers.batch_normalization(conv3d(refencode2_3, 64, 64, 3, 1, 'encode2_4') + refencode2_2, training=True,
                                          name='bnen2_4'))

        refencode3_1 = max_pool3d(refencode2_4)
        refencode3_2 = relu(
            tf.layers.batch_normalization(conv3d(refencode3_1, 64, 128, 3, 1, 'encode3_2'), training=True,
                                          name='bnen3_2'))
        refencode3_3 = relu(
            tf.layers.batch_normalization(conv3d(refencode3_2, 128, 128, 3, 1, 'encode3_3'), training=True,
                                          name='bnen3_3'))
        refencode3_4 = relu(
            tf.layers.batch_normalization(conv3d(refencode3_3, 128, 128, 3, 1, 'encode3_4') + refencode3_2,
                                          training=True,
                                          name='bnen3_4'))

        refencode4_1 = max_pool3d(refencode3_4)
        refencode4_3 = relu(
            tf.layers.batch_normalization(conv3d(refencode4_1, 128, 256, 3, 1, 'encode4_3'), training=True,
                                          name='bnen4_3'))
        refencode4_4 = relu(
            tf.layers.batch_normalization(conv3d(refencode4_3, 256, 256, 3, 1, 'encode4_4'), training=True,
                                          name='bnen4_4'))
        refencode4_5 = relu(
            tf.layers.batch_normalization(conv3d(refencode4_4, 256, 256, 3, 1, 'encode4_5') + refencode4_3,
                                          training=True,
                                          name='bnen4_5'))

        refdecode3_1 = relu(
            tf.layers.batch_normalization(deconv3d(refencode4_5, 256, 128, 3, 2, 'decode3_1'), training=True,
                                          name='bnde3_1'))
        refdecode3_2 = tf.concat([refdecode3_1, refencode3_4], 4)
        refdecode3_4 = relu(
            tf.layers.batch_normalization(conv3d(refdecode3_2, 256, 128, 3, 1, 'decode3_4'), training=True,
                                          name='bnde3_4'))
        refdecode3_5 = relu(
            tf.layers.batch_normalization(conv3d(refdecode3_4, 128, 128, 3, 1, 'decode3_5'), training=True,
                                          name='bnde3_5'))
        refdecode3_6 = relu(
            tf.layers.batch_normalization(conv3d(refdecode3_5, 128, 128, 3, 1, 'decode3_6') + refdecode3_4,
                                          training=True,
                                          name='bnde3_6'))

        refdecode4_1 = relu(
            tf.layers.batch_normalization(deconv3d(refdecode3_6, 128, 64, 3, 2, 'decode2_1'), training=True,
                                          name='bnde2_1'))
        refdecode4_2 = tf.concat([refdecode4_1, refencode2_4], 4)
        refdecode4_4 = relu(
            tf.layers.batch_normalization(conv3d(refdecode4_2, 128, 64, 3, 1, 'decode4_4'), training=True,
                                          name='bnde4_4'))
        refdecode4_5 = relu(
            tf.layers.batch_normalization(conv3d(refdecode4_4, 64, 64, 3, 1, 'decode4_5'), training=True,
                                          name='bnde4_5'))
        refdecode4_6 = relu(
            tf.layers.batch_normalization(conv3d(refdecode4_5, 64, 64, 3, 1, 'decode4_6') + refdecode4_4, training=True,
                                          name='bnde4_6'))

        refdecode5_1 = relu(
            tf.layers.batch_normalization(deconv3d(refdecode4_6, 64, 32, 3, 2, 'decode1_1'), training=True,
                                          name='bnde1_1'))
        refdecode5_2 = tf.concat([refdecode5_1, refencode1_3], 4)
        refdecode5_4 = relu(
            tf.layers.batch_normalization(conv3d(refdecode5_2, 64, 32, 3, 1, 'decode5_4'), training=True,
                                          name='bnde5_4'))
        refdecode5_5 = relu(
            tf.layers.batch_normalization(conv3d(refdecode5_4, 32, 32, 3, 1, 'decode5_5'), training=True,
                                          name='bnde5_5'))
        refdecode5_6 = relu(
            tf.layers.batch_normalization(conv3d(refdecode5_5, 32, 32, 3, 1, 'decode5_6') + refdecode5_4, training=True,
                                          name='bnde5_6'))
        refcosout = conv3d_b(refdecode5_6, 32, 1, 3, 1, 'out') + cosout
    return cosout, refcosout

def synthesis_newarc(maginput,fieldinput,tkdinput):
    with tf.variable_scope('syn'):
        #input2=tf.concat([maginput,fieldinput,tkdinput],4)
        # input2_112= tf.reshape(input2,[32,32,32])
        # input2_56 = tf.image.resize_images(input2, [16, 16,16])
        # input2_28 = tf.image.resize_images(input2, [8, 8,8])
        magconv1=relu(tf.layers.batch_normalization(conv3d(maginput,1,16,5,1,'magconv1'), training=True, name='bnmag1'))
        magconv2_1 = relu(tf.layers.batch_normalization(conv3d(magconv1, 16, 16, 3, 1, 'magconv2_1'), training=True, name='bnmag2_1'))
        magconv2_2 = relu(tf.layers.batch_normalization(conv3d(magconv2_1, 16, 16, 3, 1, 'magconv2_2')+magconv1, training=True, name='bnmag2_2'))
        magconv3_1 = relu(tf.layers.batch_normalization(conv3d(magconv2_2, 16, 16, 3, 1, 'magconv3_1'), training=True, name='bnmag3_1'))
        magconv3_2 = relu(tf.layers.batch_normalization(conv3d(magconv3_1, 16, 16, 3, 1, 'magconv3_2') + magconv2_2, training=True, name='bnmag3_2'))
        magconv4_1 = relu(tf.layers.batch_normalization(conv3d(magconv3_2, 16, 16, 3, 1, 'magconv4_1'), training=True, name='bnmag4_1'))
        magconv4_2 = relu(tf.layers.batch_normalization(conv3d(magconv4_1, 16, 16, 3, 1, 'magconv4_2') + magconv3_2, training=True, name='bnmag4_2'))
        magconv5_1 = relu(tf.layers.batch_normalization(conv3d(magconv4_2, 16, 16, 3, 1, 'magconv5_1'), training=True, name='bnmag5_1'))
        magconv5_2 = relu(tf.layers.batch_normalization(conv3d(magconv5_1, 16, 16, 3, 1, 'magconv5_2') + magconv4_2, training=True, name='bnmag5_2'))
        magconv6_1 = relu(tf.layers.batch_normalization(conv3d(magconv5_2, 16, 16, 3, 2, 'magconv6_1'), training=True,name='bnmag6_1'))
        magconv6_2 = relu(tf.layers.batch_normalization(conv3d(magconv6_1, 16, 16, 3, 1, 'magconv6_2'), training=True,name='bnmag6_2'))
        magconv7_1 = relu(tf.layers.batch_normalization(conv3d(magconv6_2, 16, 16, 3, 2, 'magconv7_1'), training=True,name='bnmag7_1'))



        tkdconv1 = relu(
            tf.layers.batch_normalization(conv3d(tkdinput, 1, 16, 5, 1, 'tkdconv1'), training=True, name='bntkd1'))
        tkdconv2_1 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv1, 16, 16, 3, 1, 'tkdconv2_1'), training=True, name='bntkd2_1'))
        tkdconv2_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv2_1, 16, 16, 3, 1, 'tkdconv2_2') + tkdconv1, training=True,
                                          name='bntkd2_2'))
        tkdconv3_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv2_2, 16, 16, 3, 1, 'tkdconv3_1'), training=True,
                                                        name='bntkd3_1'))
        tkdconv3_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv3_1, 16, 16, 3, 1, 'tkdconv3_2') + tkdconv2_2, training=True,
                                          name='bntkd3_2'))
        tkdconv4_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv3_2, 16, 16, 3, 1, 'tkdconv4_1'), training=True,
                                                        name='bntkd4_1'))
        tkdconv4_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv4_1, 16, 16, 3, 1, 'tkdconv4_2') + tkdconv3_2, training=True,
                                          name='bntkd4_2'))
        tkdconv5_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv4_2, 16, 16, 3, 1, 'tkdonv5_1'), training=True,
                                                        name='bntkd5_1'))
        tkdconv5_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv5_1, 16, 16, 3, 1, 'tkdconv5_2') + tkdconv4_2, training=True,
                                          name='bntkd5_2'))
        tkdconv6_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv5_2, 16, 16, 3, 2, 'tkdconv6_1'), training=True,name='bntkd6_1'))
        tkdconv6_2 = relu(tf.layers.batch_normalization(conv3d(tkdconv6_1, 16, 16, 3, 1, 'tkdconv6_2'), training=True,name='bntkd6_2'))
        tkdconv7_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv6_2, 16, 16, 3, 2, 'tkdconv7_1'), training=True,name='bntkd7_1'))



        fieldconv1=relu(tf.layers.batch_normalization(conv3d(fieldinput,1,16,5,1,'fieldconv1'), training=True, name='bnfield1'))
        fieldconv2_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv1, 16, 16, 3, 1, 'fieldconv2_1'), training=True, name='bndield2_1'))
        fieldconv2_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_1, 16, 16, 3, 1, 'fieldconv2_2')+fieldconv1, training=True, name='bnfield2_2'))
        fieldconv3_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_2, 16, 16, 3, 1, 'fieldconv3_1'), training=True, name='bnfield3_1'))
        fieldconv3_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_1, 16, 16, 3, 1, 'fieldconv3_2') + fieldconv2_2, training=True, name='bnfield3_2'))
        fieldconv4_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_2, 16, 16, 3, 1, 'fieldconv4_1'), training=True, name='bnfield4_1'))
        fieldconv4_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_1, 16, 16, 3, 1, 'fieldconv4_2') + fieldconv3_2, training=True, name='bnfield4_2'))
        fieldconv5_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_2, 16, 16, 3, 1, 'fieldconv5_1'), training=True, name='bnfield5_1'))
        fieldconv5_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv5_1, 16, 16, 3, 1, 'fieldconv5_2') + fieldconv4_2, training=True, name='bnfield5_2'))
        fieldconv6_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv5_2, 16, 16, 3, 2, 'fieldconv6_1'), training=True,name='bnfield6_1'))
        fieldconv6_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv6_1, 16, 16, 3, 1, 'fieldconv6_2'), training=True,name='bnfield6_2'))
        fieldconv7_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv6_2, 16, 16, 3, 2, 'fieldconv7_1'), training=True,name='bnfield7_1'))



        concat7_1=tf.concat([magconv7_1,tkdconv7_1,fieldconv7_1],4)
        globalpool=tf.reduce_mean(concat7_1,[2,3],keep_dims=True)
        fusionflatten = tf.reshape(globalpool, (4, 192))
        magdense = tf.layers.dense(fusionflatten, 256)
        tkddense = tf.layers.dense(fusionflatten, 256)
        fielddense = tf.layers.dense(fusionflatten, 256)

        magweight = (tf.reshape(magdense, (4, 16, 1, 1, 16)))
        magin = magconv5_2 * magweight
        tkdweight = (tf.reshape(tkddense, (4, 16, 1, 1, 16)))
        tkdin = tkdconv5_2 * tkdweight
        fieldweight = (tf.reshape(fielddense, (4, 16, 1, 1, 16)))
        fieldin = fieldconv5_2 * fieldweight

        encodeinput=tf.concat([magin,fieldin,tkdin],4)
        encode1_1=relu(tf.layers.batch_normalization(conv3d(encodeinput,48,32,3,1,'encode1_1'), training=True, name='bnen1'))
        encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=True, name='bnen1_2'))
        encode1_3 = relu(tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 1, 'encode1_3')+encode1_1, training=True, name='bnen1_3'))

        encode2_1=max_pool3d(encode1_3)
        encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=True, name='bnen2_2'))
        encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=True, name='bnen2_3'))
        encode2_4 = relu(tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 1, 'encode2_4')+encode2_2, training=True, name='bnen2_4'))

        encode3_1=max_pool3d(encode2_4)
        encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=True, name='bnen3_2'))
        encode3_3 = relu(tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=True, name='bnen3_3'))
        encode3_4 = relu(tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 1, 'encode3_4')+encode3_2, training=True, name='bnen3_4'))

        encode4_1 = max_pool3d(encode3_4)
        encode4_3 = relu(tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_3'), training=True, name='bnen4_3'))
        encode4_4 = relu(tf.layers.batch_normalization(conv3d(encode4_3, 256, 256, 3, 1, 'encode4_4'), training=True, name='bnen4_4'))
        encode4_5 = relu(tf.layers.batch_normalization(conv3d(encode4_4, 256, 256, 3, 1, 'encode4_5')+encode4_3, training=True, name='bnen4_5'))

        decode3_1 = relu(tf.layers.batch_normalization(deconv3d(encode4_5, 256, 128, 3, 2, 'decode3_1'), training=True,
                                                       name='bnde3_1'))
        decode3_2=tf.concat([decode3_1,encode3_4],4)
        decode3_4 = relu(tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=True, name='bnde3_4'))
        decode3_5 = relu(tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_5'), training=True, name='bnde3_5'))
        decode3_6 = relu(tf.layers.batch_normalization(conv3d(decode3_5, 128, 128, 3, 1, 'decode3_6')+decode3_4, training=True, name='bnde3_6'))

        decode4_1 = relu(tf.layers.batch_normalization(deconv3d(decode3_6, 128, 64, 3, 2, 'decode2_1'), training=True,
                                          name='bnde2_1'))
        decode4_2 = tf.concat([decode4_1,encode2_4],4)
        decode4_4 = relu(tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=True, name='bnde4_4'))
        decode4_5 = relu(tf.layers.batch_normalization(conv3d(decode4_4, 64, 64, 3, 1, 'decode4_5'), training=True, name='bnde4_5'))
        decode4_6 = relu(tf.layers.batch_normalization(conv3d(decode4_5, 64, 64, 3, 1, 'decode4_6')+decode4_4, training=True, name='bnde4_6'))

        decode5_1 = relu(tf.layers.batch_normalization(deconv3d(decode4_6, 64, 32, 3, 2, 'decode1_1'), training=True,
                                                       name='bnde1_1'))
        decode5_2 = tf.concat([decode5_1,encode1_3],4)
        decode5_4 = relu(tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=True, name='bnde5_4'))
        decode5_5 = relu(tf.layers.batch_normalization(conv3d(decode5_4, 32, 32, 3, 1, 'decode5_5'), training=True, name='bnde5_5'))
        decode5_6 = relu(tf.layers.batch_normalization(conv3d(decode5_5, 32, 32, 3, 1, 'decode5_6')+decode5_4, training=True, name='bnde5_6'))
        cosout=conv3d_b(decode5_6, 32, 1, 3, 1, 'out')+tkdinput


        return cosout
def synthesis_se(maginput,fieldinput,tkdinput):
    with tf.variable_scope('syn'):
        #input2=tf.concat([maginput,fieldinput,tkdinput],4)
        # input2_112= tf.reshape(input2,[32,32,32])
        # input2_56 = tf.image.resize_images(input2, [16, 16,16])
        # input2_28 = tf.image.resize_images(input2, [8, 8,8])
        magconv1=relu(tf.layers.batch_normalization(conv3d(maginput,1,16,5,1,'magconv1'), training=True, name='bnmag1'))
        magconv2_1 = relu(tf.layers.batch_normalization(conv3d(magconv1, 16, 16, 3, 1, 'magconv2_1'), training=True, name='bnmag2_1'))
        magconv2_2 = relu(tf.layers.batch_normalization(conv3d(magconv2_1, 16, 16, 3, 1, 'magconv2_2')+magconv1, training=True, name='bnmag2_2'))
        magconv3_1 = relu(tf.layers.batch_normalization(conv3d(magconv2_2, 16, 16, 3, 1, 'magconv3_1'), training=True, name='bnmag3_1'))
        magconv3_2 = relu(tf.layers.batch_normalization(conv3d(magconv3_1, 16, 16, 3, 1, 'magconv3_2') + magconv2_2, training=True, name='bnmag3_2'))
        magconv4_1 = relu(tf.layers.batch_normalization(conv3d(magconv3_2, 16, 16, 3, 1, 'magconv4_1'), training=True, name='bnmag4_1'))
        magconv4_2 = relu(tf.layers.batch_normalization(conv3d(magconv4_1, 16, 16, 3, 1, 'magconv4_2') + magconv3_2, training=True, name='bnmag4_2'))
        magconv5_1 = relu(tf.layers.batch_normalization(conv3d(magconv4_2, 16, 16, 3, 1, 'magconv5_1'), training=True, name='bnmag5_1'))
        magconv5_2 = relu(tf.layers.batch_normalization(conv3d(magconv5_1, 16, 16, 3, 1, 'magconv5_2') + magconv4_2, training=True, name='bnmag5_2'))
        magconv6_1 = relu(tf.layers.batch_normalization(conv3d(magconv5_2, 16, 16, 3, 2, 'magconv6_1'), training=True,name='bnmag6_1'))
        magconv6_2 = relu(tf.layers.batch_normalization(conv3d(magconv6_1, 16, 16, 3, 1, 'magconv6_2'), training=True,name='bnmag6_2'))
        magconv7_1 = relu(tf.layers.batch_normalization(conv3d(magconv6_2, 16, 16, 3, 2, 'magconv7_1'), training=True,name='bnmag7_1'))
        magglobalpool = tf.reduce_mean(magconv7_1, [2, 3], keep_dims=True)
        magflatten = tf.reshape(magglobalpool, (4, 64))
        magdense = relu(tf.layers.dense(magflatten, 256))
        magdense1 = tf.sigmoid(tf.layers.dense(magdense, 256))
        magweight = (tf.reshape(magdense1, (4, 16, 1, 1, 16)))
        magin = magconv5_2 * magweight

        tkdconv1 = relu(
            tf.layers.batch_normalization(conv3d(tkdinput, 1, 16, 5, 1, 'tkdconv1'), training=True, name='bntkd1'))
        tkdconv2_1 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv1, 16, 16, 3, 1, 'tkdconv2_1'), training=True, name='bntkd2_1'))
        tkdconv2_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv2_1, 16, 16, 3, 1, 'tkdconv2_2') + tkdconv1, training=True,
                                          name='bntkd2_2'))
        tkdconv3_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv2_2, 16, 16, 3, 1, 'tkdconv3_1'), training=True,
                                                        name='bntkd3_1'))
        tkdconv3_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv3_1, 16, 16, 3, 1, 'tkdconv3_2') + tkdconv2_2, training=True,
                                          name='bntkd3_2'))
        tkdconv4_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv3_2, 16, 16, 3, 1, 'tkdconv4_1'), training=True,
                                                        name='bntkd4_1'))
        tkdconv4_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv4_1, 16, 16, 3, 1, 'tkdconv4_2') + tkdconv3_2, training=True,
                                          name='bntkd4_2'))
        tkdconv5_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv4_2, 16, 16, 3, 1, 'tkdonv5_1'), training=True,
                                                        name='bntkd5_1'))
        tkdconv5_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv5_1, 16, 16, 3, 1, 'tkdconv5_2') + tkdconv4_2, training=True,
                                          name='bntkd5_2'))
        tkdconv6_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv5_2, 16, 16, 3, 2, 'tkdconv6_1'), training=True,name='bntkd6_1'))
        tkdconv6_2 = relu(tf.layers.batch_normalization(conv3d(tkdconv6_1, 16, 16, 3, 1, 'tkdconv6_2'), training=True,name='bntkd6_2'))
        tkdconv7_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv6_2, 16, 16, 3, 2, 'tkdconv7_1'), training=True,name='bntkd7_1'))
        tkdglobalpool = tf.reduce_mean(tkdconv7_1, [2, 3], keep_dims=True)
        tkdflatten = tf.reshape(tkdglobalpool, (4, 64))
        tkddense = relu(tf.layers.dense(tkdflatten, 256))
        tkddense1 = tf.sigmoid(tf.layers.dense(tkddense, 256))
        tkdweight = (tf.reshape(tkddense1, (4, 16, 1, 1, 16)))
        tkdin = tkdconv5_2 * tkdweight


        fieldconv1=relu(tf.layers.batch_normalization(conv3d(fieldinput,1,16,5,1,'fieldconv1'), training=True, name='bnfield1'))
        fieldconv2_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv1, 16, 16, 3, 1, 'fieldconv2_1'), training=True, name='bndield2_1'))
        fieldconv2_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_1, 16, 16, 3, 1, 'fieldconv2_2')+fieldconv1, training=True, name='bnfield2_2'))
        fieldconv3_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_2, 16, 16, 3, 1, 'fieldconv3_1'), training=True, name='bnfield3_1'))
        fieldconv3_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_1, 16, 16, 3, 1, 'fieldconv3_2') + fieldconv2_2, training=True, name='bnfield3_2'))
        fieldconv4_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_2, 16, 16, 3, 1, 'fieldconv4_1'), training=True, name='bnfield4_1'))
        fieldconv4_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_1, 16, 16, 3, 1, 'fieldconv4_2') + fieldconv3_2, training=True, name='bnfield4_2'))
        fieldconv5_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_2, 16, 16, 3, 1, 'fieldconv5_1'), training=True, name='bnfield5_1'))
        fieldconv5_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv5_1, 16, 16, 3, 1, 'fieldconv5_2') + fieldconv4_2, training=True, name='bnfield5_2'))
        fieldconv6_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv5_2, 16, 16, 3, 2, 'fieldconv6_1'), training=True,name='bnfield6_1'))
        fieldconv6_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv6_1, 16, 16, 3, 1, 'fieldconv6_2'), training=True,name='bnfield6_2'))
        fieldconv7_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv6_2, 16, 16, 3, 2, 'fieldconv7_1'), training=True,name='bnfield7_1'))
        fieldglobalpool = tf.reduce_mean(fieldconv7_1, [2, 3], keep_dims=True)
        fieldflatten = tf.reshape(fieldglobalpool, (4, 64))
        fielddense = relu(tf.layers.dense(fieldflatten, 256))
        fielddense1 = tf.sigmoid(tf.layers.dense(fielddense, 256))
        fieldweight = (tf.reshape(fielddense1, (4, 16, 1, 1, 16)))
        fieldin = fieldconv5_2 * fieldweight




        encodeinput=tf.concat([magin,fieldin,tkdin],4)
        encode1_1=relu(tf.layers.batch_normalization(conv3d(encodeinput,48,32,3,1,'encode1_1'), training=True, name='bnen1'))
        encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=True, name='bnen1_2'))
        encode1_3 = relu(tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 1, 'encode1_3')+encode1_1, training=True, name='bnen1_3'))

        encode2_1=max_pool3d(encode1_3)
        encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=True, name='bnen2_2'))
        encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=True, name='bnen2_3'))
        encode2_4 = relu(tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 1, 'encode2_4')+encode2_2, training=True, name='bnen2_4'))

        encode3_1=max_pool3d(encode2_4)
        encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=True, name='bnen3_2'))
        encode3_3 = relu(tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=True, name='bnen3_3'))
        encode3_4 = relu(tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 1, 'encode3_4')+encode3_2, training=True, name='bnen3_4'))

        encode4_1 = max_pool3d(encode3_4)
        encode4_3 = relu(tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_3'), training=True, name='bnen4_3'))
        encode4_4 = relu(tf.layers.batch_normalization(conv3d(encode4_3, 256, 256, 3, 1, 'encode4_4'), training=True, name='bnen4_4'))
        encode4_5 = relu(tf.layers.batch_normalization(conv3d(encode4_4, 256, 256, 3, 1, 'encode4_5')+encode4_3, training=True, name='bnen4_5'))

        decode3_1 = relu(tf.layers.batch_normalization(deconv3d(encode4_5, 256, 128, 3, 2, 'decode3_1'), training=True,
                                                       name='bnde3_1'))
        decode3_2=tf.concat([decode3_1,encode3_4],4)
        decode3_4 = relu(tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=True, name='bnde3_4'))
        decode3_5 = relu(tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_5'), training=True, name='bnde3_5'))
        decode3_6 = relu(tf.layers.batch_normalization(conv3d(decode3_5, 128, 128, 3, 1, 'decode3_6')+decode3_4, training=True, name='bnde3_6'))

        decode4_1 = relu(tf.layers.batch_normalization(deconv3d(decode3_6, 128, 64, 3, 2, 'decode2_1'), training=True,
                                          name='bnde2_1'))
        decode4_2 = tf.concat([decode4_1,encode2_4],4)
        decode4_4 = relu(tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=True, name='bnde4_4'))
        decode4_5 = relu(tf.layers.batch_normalization(conv3d(decode4_4, 64, 64, 3, 1, 'decode4_5'), training=True, name='bnde4_5'))
        decode4_6 = relu(tf.layers.batch_normalization(conv3d(decode4_5, 64, 64, 3, 1, 'decode4_6')+decode4_4, training=True, name='bnde4_6'))

        decode5_1 = relu(tf.layers.batch_normalization(deconv3d(decode4_6, 64, 32, 3, 2, 'decode1_1'), training=True,
                                                       name='bnde1_1'))
        decode5_2 = tf.concat([decode5_1,encode1_3],4)
        decode5_4 = relu(tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=True, name='bnde5_4'))
        decode5_5 = relu(tf.layers.batch_normalization(conv3d(decode5_4, 32, 32, 3, 1, 'decode5_5'), training=True, name='bnde5_5'))
        decode5_6 = relu(tf.layers.batch_normalization(conv3d(decode5_5, 32, 32, 3, 1, 'decode5_6')+decode5_4, training=True, name='bnde5_6'))
        cosout=conv3d_b(decode5_6, 32, 1, 3, 1, 'out')+tkdinput


        return cosout
def synthesis_nogp(maginput,fieldinput,tkdinput):
    with tf.variable_scope('syn'):
        #input2=tf.concat([maginput,fieldinput,tkdinput],4)
        # input2_112= tf.reshape(input2,[32,32,32])
        # input2_56 = tf.image.resize_images(input2, [16, 16,16])
        # input2_28 = tf.image.resize_images(input2, [8, 8,8])
        magconv1=relu(tf.layers.batch_normalization(conv3d(maginput,1,48,3,1,'magconv1'), training=True, name='bnmag1'))
        magconv2_33 = relu(tf.layers.batch_normalization(conv3d(magconv1, 48, 16, 3, 1, 'magconv2_33'), training=True, name='bnmag2_33'))
        magconv2_11 = relu(tf.layers.batch_normalization(conv3d(magconv1, 48, 16, 1, 1, 'magconv2_11'), training=True,name='bnmag2_11'))
        magconv2_55 = relu(tf.layers.batch_normalization(conv3d(magconv1, 48, 16, 5, 1, 'magconv2_55'), training=True,name='bnmag2_55'))
        magcat1=tf.concat([magconv2_33,magconv2_11,magconv2_55],4)

        magconv3_33 = relu(tf.layers.batch_normalization(conv3d(magcat1, 48, 16, 3, 1, 'magconv3_33'), training=True,name='bnmag3_33'))
        magconv3_11 = relu(tf.layers.batch_normalization(conv3d(magcat1, 48, 16, 1, 1, 'magconv3_11'), training=True,name='bnmag3_11'))
        magconv3_55 = relu(tf.layers.batch_normalization(conv3d(magcat1, 48, 16, 5, 1, 'magconv3_55'), training=True,name='bnmag3_55'))
        magcat2 = tf.concat([magconv3_33, magconv3_11, magconv3_55], 4)

        magconv4_33 = relu(tf.layers.batch_normalization(conv3d(magcat2, 48, 16, 3, 1, 'magconv4_33'), training=True,name='bnmag4_33'))
        magconv4_11 = relu(tf.layers.batch_normalization(conv3d(magcat2, 48, 16, 1, 1, 'magconv4_11'), training=True,name='bnmag4_11'))
        magconv4_55 = relu(tf.layers.batch_normalization(conv3d(magcat2, 48, 16, 5, 1, 'magconv4_55'), training=True,name='bnmag4_55'))
        magcat3 = tf.concat([magconv4_33, magconv4_11, magconv4_55], 4)

        # magconv5_33 = relu(tf.layers.batch_normalization(conv3d(magcat3, 48, 16, 3, 1, 'magconv5_33'), training=True,name='bnmag5_33'))
        # magconv5_11 = relu(tf.layers.batch_normalization(conv3d(magcat3, 48, 16, 1, 1, 'magconv5_11'), training=True,name='bnmag5_11'))
        # magconv5_55 = relu(tf.layers.batch_normalization(conv3d(magcat3, 48, 16, 5, 1, 'magconv5_55'), training=True,name='bnmag5_55'))
        # magcat4 = tf.concat([magconv5_33, magconv5_11, magconv5_55], 4)


        magconcat=tf.concat([magcat1,magcat2,magcat3],4)
        magweight=tf.get_variable('magw', shape=[144], dtype=tf.float32, initializer=tf.ones_initializer(),regularizer=None, trainable=True, collections=None)
        magzong=magconcat*magweight
        magin=relu(tf.layers.batch_normalization(conv3d(magzong, 144, 32, 3, 1, 'magin') , training=True, name='bnmagin'))

        tkdconv1 = relu(tf.layers.batch_normalization(conv3d(tkdinput, 1, 48, 3, 1, 'tkdconv1'), training=True, name='bntkd1'))
        tkdconv2_33 = relu(tf.layers.batch_normalization(conv3d(tkdconv1, 48, 16, 3, 1, 'tkdconv2_33'), training=True,name='bntkd2_33'))
        tkdconv2_11 = relu(tf.layers.batch_normalization(conv3d(tkdconv1, 48, 16, 1, 1, 'tkdconv2_11'), training=True,name='bntkd2_11'))
        tkdconv2_55 = relu(tf.layers.batch_normalization(conv3d(tkdconv1, 48, 16, 5, 1, 'tkdconv2_55'), training=True,name='bntkd2_55'))
        tkdcat1 = tf.concat([tkdconv2_33, tkdconv2_11, tkdconv2_55], 4)

        tkdconv3_33 = relu(tf.layers.batch_normalization(conv3d(tkdcat1, 48, 16, 3, 1, 'tkdconv3_33'), training=True,name='bntkd3_33'))
        tkdconv3_11 = relu(tf.layers.batch_normalization(conv3d(tkdcat1, 48, 16, 1, 1, 'tkdconv3_11'), training=True, name='bntkd3_11'))
        tkdconv3_55 = relu(tf.layers.batch_normalization(conv3d(tkdcat1, 48, 16, 5, 1, 'tkdconv3_55'), training=True,name='bntkd3_55'))
        tkdcat2 = tf.concat([tkdconv3_33, tkdconv3_11, tkdconv3_55], 4)

        tkdconv4_33 = relu(tf.layers.batch_normalization(conv3d(tkdcat2, 48, 16, 3, 1, 'tkdconv4_33'), training=True,name='bntkd4_33'))
        tkdconv4_11 = relu(tf.layers.batch_normalization(conv3d(tkdcat2, 48, 16, 1, 1, 'tkdconv4_11'), training=True,name='bntkd4_11'))
        tkdconv4_55 = relu(tf.layers.batch_normalization(conv3d(tkdcat2, 48, 16, 5, 1, 'tkdconv4_55'), training=True,name='bntkd4_55'))
        tkdcat3 = tf.concat([tkdconv4_33, tkdconv4_11, tkdconv4_55], 4)

        # tkdconv5_33 = relu(tf.layers.batch_normalization(conv3d(tkdcat3, 48, 16, 3, 1, 'tkdconv5_33'), training=True,name='bntkd5_33'))
        # tkdconv5_11 = relu(tf.layers.batch_normalization(conv3d(tkdcat3, 48, 16, 1, 1, 'tkdconv5_11'), training=True,name='bntkd5_11'))
        # tkdconv5_55 = relu(tf.layers.batch_normalization(conv3d(tkdcat3, 48, 16, 5, 1, 'tkdconv5_55'), training=True,name='bntkd5_55'))
        # tkdcat4 = tf.concat([tkdconv5_33, tkdconv5_11, tkdconv5_55], 4)

        tkdconcat = tf.concat([tkdcat1, tkdcat2, tkdcat3], 4)
        tkdweight = tf.get_variable('tkdw', shape=[144], dtype=tf.float32,initializer=tf.ones_initializer(),regularizer=None, trainable=True, collections=None)
        tkdzong = tkdconcat * tkdweight
        tkdin = relu(tf.layers.batch_normalization(conv3d(tkdzong, 144, 32, 3, 1, 'tkdin'), training=True, name='bntkdin'))


        fieldconv1 = relu(tf.layers.batch_normalization(conv3d(fieldinput, 1, 48, 3, 1, 'fieldconv1'), training=True, name='bnfield1'))
        fieldconv2_33 = relu(tf.layers.batch_normalization(conv3d(fieldconv1, 48, 16, 3, 1, 'fieldconv2_33'), training=True,name='bnfield2_33'))
        fieldconv2_11 = relu(tf.layers.batch_normalization(conv3d(fieldconv1, 48, 16, 1, 1, 'fieldconv2_11'), training=True,name='bnfield2_11'))
        fieldconv2_55 = relu(tf.layers.batch_normalization(conv3d(fieldconv1, 48, 16, 5, 1, 'fieldconv2_55'), training=True,name='bnfield2_55'))
        fieldcat1 = tf.concat([fieldconv2_33, fieldconv2_11, fieldconv2_55], 4)

        fieldconv3_33 = relu(tf.layers.batch_normalization(conv3d(fieldcat1, 48, 16, 3, 1, 'fieldconv3_33'), training=True,name='bnfield3_33'))
        fieldconv3_11 = relu(tf.layers.batch_normalization(conv3d(fieldcat1, 48, 16, 1, 1, 'fieldconv3_11'), training=True,name='bnfield3_11'))
        fieldconv3_55 = relu(tf.layers.batch_normalization(conv3d(fieldcat1, 48, 16, 5, 1, 'fieldconv3_55'), training=True,name='bnfield3_55'))
        fieldcat2 = tf.concat([fieldconv3_33, fieldconv3_11, fieldconv3_55], 4)

        fieldconv4_33 = relu(tf.layers.batch_normalization(conv3d(fieldcat2, 48, 16, 3, 1, 'fieldconv4_33'), training=True,name='bnfield4_33'))
        fieldconv4_11 = relu(tf.layers.batch_normalization(conv3d(fieldcat2, 48, 16, 1, 1, 'fieldconv4_11'), training=True,name='bnfield4_11'))
        fieldconv4_55 = relu(tf.layers.batch_normalization(conv3d(fieldcat2, 48, 16, 5, 1, 'fieldconv4_55'), training=True,name='bnfield4_55'))
        fieldcat3 = tf.concat([fieldconv4_33, fieldconv4_11, fieldconv4_55], 4)

        # fieldconv5_33 = relu(tf.layers.batch_normalization(conv3d(fieldcat3, 48, 16, 3, 1, 'fieldconv5_33'), training=True,name='bnfield5_33'))
        # fieldconv5_11 = relu(tf.layers.batch_normalization(conv3d(fieldcat3, 48, 16, 1, 1, 'fieldconv5_11'), training=True,name='bnfield5_11'))
        # fieldconv5_55 = relu(tf.layers.batch_normalization(conv3d(fieldcat3, 48, 16, 5, 1, 'fieldconv5_55'), training=True,name='bnfield5_55'))
        # fieldcat4 = tf.concat([fieldconv5_33, fieldconv5_11, fieldconv5_55], 4)

        fieldconcat = tf.concat([fieldcat1, fieldcat2, fieldcat3], 4)
        fieldweight = tf.get_variable('fieldw', shape=[144], dtype=tf.float32,initializer=tf.ones_initializer(), regularizer=None,trainable=True, collections=None)
        fieldzong = fieldconcat * fieldweight
        fieldin = relu(tf.layers.batch_normalization(conv3d(fieldzong, 144, 32, 3, 1, 'fieldin'), training=True, name='bnfieldin'))

        encodeinput=tf.concat([magin,fieldin,tkdin],4)
        encode1_1=relu(tf.layers.batch_normalization(conv3d(encodeinput,96,32,3,1,'encode1_1'), training=True, name='bnen1'))
        encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=True, name='bnen1_2'))
        encode1_3 = relu(tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 1, 'encode1_3')+encode1_1, training=True, name='bnen1_3'))

        encode2_1=max_pool3d(encode1_3)
        encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=True, name='bnen2_2'))
        encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=True, name='bnen2_3'))
        encode2_4 = relu(tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 1, 'encode2_4')+encode2_2, training=True, name='bnen2_4'))

        encode3_1=max_pool3d(encode2_4)
        encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=True, name='bnen3_2'))
        encode3_3 = relu(tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=True, name='bnen3_3'))
        encode3_4 = relu(tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 1, 'encode3_4')+encode3_2, training=True, name='bnen3_4'))

        encode4_1 = max_pool3d(encode3_4)
        encode4_3 = relu(tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_3'), training=True, name='bnen4_3'))
        encode4_4 = relu(tf.layers.batch_normalization(conv3d(encode4_3, 256, 256, 3, 1, 'encode4_4'), training=True, name='bnen4_4'))
        encode4_5 = relu(tf.layers.batch_normalization(conv3d(encode4_4, 256, 256, 3, 1, 'encode4_5')+encode4_3, training=True, name='bnen4_5'))

        decode3_1 = relu(tf.layers.batch_normalization(deconv3d(encode4_5, 256, 128, 3, 2, 'decode3_1'), training=True,
                                                       name='bnde3_1'))
        decode3_2=tf.concat([decode3_1,encode3_4],4)
        decode3_4 = relu(tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=True, name='bnde3_4'))
        decode3_5 = relu(tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_5'), training=True, name='bnde3_5'))
        decode3_6 = relu(tf.layers.batch_normalization(conv3d(decode3_5, 128, 128, 3, 1, 'decode3_6')+decode3_4, training=True, name='bnde3_6'))

        decode4_1 = relu(tf.layers.batch_normalization(deconv3d(decode3_6, 128, 64, 3, 2, 'decode2_1'), training=True,
                                          name='bnde2_1'))
        decode4_2 = tf.concat([decode4_1,encode2_4],4)
        decode4_4 = relu(tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=True, name='bnde4_4'))
        decode4_5 = relu(tf.layers.batch_normalization(conv3d(decode4_4, 64, 64, 3, 1, 'decode4_5'), training=True, name='bnde4_5'))
        decode4_6 = relu(tf.layers.batch_normalization(conv3d(decode4_5, 64, 64, 3, 1, 'decode4_6')+decode4_4, training=True, name='bnde4_6'))

        decode5_1 = relu(tf.layers.batch_normalization(deconv3d(decode4_6, 64, 32, 3, 2, 'decode1_1'), training=True,
                                                       name='bnde1_1'))
        decode5_2 = tf.concat([decode5_1,encode1_3],4)
        decode5_4 = relu(tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=True, name='bnde5_4'))
        decode5_5 = relu(tf.layers.batch_normalization(conv3d(decode5_4, 32, 32, 3, 1, 'decode5_5'), training=True, name='bnde5_5'))
        decode5_6 = relu(tf.layers.batch_normalization(conv3d(decode5_5, 32, 32, 3, 1, 'decode5_6')+decode5_4, training=True, name='bnde5_6'))
        cosout=conv3d_b(decode5_6, 32, 1, 3, 1, 'out')+tkdinput


        return cosout
def t_synthesis_se(maginput,fieldinput,tkdinput):
    with tf.variable_scope('syn'):
        #input2=tf.concat([maginput,fieldinput,tkdinput],4)
        # input2_112= tf.reshape(input2,[32,32,32])
        # input2_56 = tf.image.resize_images(input2, [16, 16,16])
        # input2_28 = tf.image.resize_images(input2, [8, 8,8])
        magconv1=relu(tf.layers.batch_normalization(conv3d(maginput,1,16,5,1,'magconv1'), training=True, name='bnmag1'))
        magconv2_1 = relu(tf.layers.batch_normalization(conv3d(magconv1, 16, 16, 3, 1, 'magconv2_1'), training=True, name='bnmag2_1'))
        magconv2_2 = relu(tf.layers.batch_normalization(conv3d(magconv2_1, 16, 16, 3, 1, 'magconv2_2')+magconv1, training=True, name='bnmag2_2'))
        magconv3_1 = relu(tf.layers.batch_normalization(conv3d(magconv2_2, 16, 16, 3, 1, 'magconv3_1'), training=True, name='bnmag3_1'))
        magconv3_2 = relu(tf.layers.batch_normalization(conv3d(magconv3_1, 16, 16, 3, 1, 'magconv3_2') + magconv2_2, training=True, name='bnmag3_2'))
        magconv4_1 = relu(tf.layers.batch_normalization(conv3d(magconv3_2, 16, 16, 3, 1, 'magconv4_1'), training=True, name='bnmag4_1'))
        magconv4_2 = relu(tf.layers.batch_normalization(conv3d(magconv4_1, 16, 16, 3, 1, 'magconv4_2') + magconv3_2, training=True, name='bnmag4_2'))
        magconv5_1 = relu(tf.layers.batch_normalization(conv3d(magconv4_2, 16, 16, 3, 1, 'magconv5_1'), training=True, name='bnmag5_1'))
        magconv5_2 = relu(tf.layers.batch_normalization(conv3d(magconv5_1, 16, 16, 3, 1, 'magconv5_2') + magconv4_2, training=True, name='bnmag5_2'))
        magconv6_1 = relu(tf.layers.batch_normalization(conv3d(magconv5_2, 16, 16, 3, 2, 'magconv6_1'), training=True,name='bnmag6_1'))
        magconv6_2 = relu(tf.layers.batch_normalization(conv3d(magconv6_1, 16, 16, 3, 1, 'magconv6_2'), training=True,name='bnmag6_2'))
        magconv7_1 = relu(tf.layers.batch_normalization(conv3d(magconv6_2, 16, 16, 3, 2, 'magconv7_1'), training=True,name='bnmag7_1'))
        magglobalpool = tf.reduce_mean(magconv7_1, [2, 3], keep_dims=True)
        magflatten = tf.reshape(magglobalpool, (1, 64))
        magdense = relu(tf.layers.dense(magflatten, 256))
        magdense1 = tf.sigmoid(tf.layers.dense(magdense, 256))
        magweight = (tf.reshape(magdense1, (1, 16, 1, 1, 16)))
        magin = magconv5_2 * magweight

        tkdconv1 = relu(
            tf.layers.batch_normalization(conv3d(tkdinput, 1, 16, 5, 1, 'tkdconv1'), training=True, name='bntkd1'))
        tkdconv2_1 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv1, 16, 16, 3, 1, 'tkdconv2_1'), training=True, name='bntkd2_1'))
        tkdconv2_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv2_1, 16, 16, 3, 1, 'tkdconv2_2') + tkdconv1, training=True,
                                          name='bntkd2_2'))
        tkdconv3_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv2_2, 16, 16, 3, 1, 'tkdconv3_1'), training=True,
                                                        name='bntkd3_1'))
        tkdconv3_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv3_1, 16, 16, 3, 1, 'tkdconv3_2') + tkdconv2_2, training=True,
                                          name='bntkd3_2'))
        tkdconv4_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv3_2, 16, 16, 3, 1, 'tkdconv4_1'), training=True,
                                                        name='bntkd4_1'))
        tkdconv4_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv4_1, 16, 16, 3, 1, 'tkdconv4_2') + tkdconv3_2, training=True,
                                          name='bntkd4_2'))
        tkdconv5_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv4_2, 16, 16, 3, 1, 'tkdonv5_1'), training=True,
                                                        name='bntkd5_1'))
        tkdconv5_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv5_1, 16, 16, 3, 1, 'tkdconv5_2') + tkdconv4_2, training=True,
                                          name='bntkd5_2'))
        tkdconv6_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv5_2, 16, 16, 3, 2, 'tkdconv6_1'), training=True,name='bntkd6_1'))
        tkdconv6_2 = relu(tf.layers.batch_normalization(conv3d(tkdconv6_1, 16, 16, 3, 1, 'tkdconv6_2'), training=True,name='bntkd6_2'))
        tkdconv7_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv6_2, 16, 16, 3, 2, 'tkdconv7_1'), training=True,name='bntkd7_1'))
        tkdglobalpool = tf.reduce_mean(tkdconv7_1, [2, 3], keep_dims=True)
        tkdflatten = tf.reshape(tkdglobalpool, (1, 64))
        tkddense = relu(tf.layers.dense(tkdflatten, 256))
        tkddense1 = tf.sigmoid(tf.layers.dense(tkddense, 256))
        tkdweight = (tf.reshape(tkddense1, (1, 16, 1, 1, 16)))
        tkdin = tkdconv5_2 * tkdweight


        fieldconv1=relu(tf.layers.batch_normalization(conv3d(fieldinput,1,16,5,1,'fieldconv1'), training=True, name='bnfield1'))
        fieldconv2_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv1, 16, 16, 3, 1, 'fieldconv2_1'), training=True, name='bndield2_1'))
        fieldconv2_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_1, 16, 16, 3, 1, 'fieldconv2_2')+fieldconv1, training=True, name='bnfield2_2'))
        fieldconv3_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_2, 16, 16, 3, 1, 'fieldconv3_1'), training=True, name='bnfield3_1'))
        fieldconv3_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_1, 16, 16, 3, 1, 'fieldconv3_2') + fieldconv2_2, training=True, name='bnfield3_2'))
        fieldconv4_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_2, 16, 16, 3, 1, 'fieldconv4_1'), training=True, name='bnfield4_1'))
        fieldconv4_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_1, 16, 16, 3, 1, 'fieldconv4_2') + fieldconv3_2, training=True, name='bnfield4_2'))
        fieldconv5_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_2, 16, 16, 3, 1, 'fieldconv5_1'), training=True, name='bnfield5_1'))
        fieldconv5_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv5_1, 16, 16, 3, 1, 'fieldconv5_2') + fieldconv4_2, training=True, name='bnfield5_2'))
        fieldconv6_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv5_2, 16, 16, 3, 2, 'fieldconv6_1'), training=True,name='bnfield6_1'))
        fieldconv6_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv6_1, 16, 16, 3, 1, 'fieldconv6_2'), training=True,name='bnfield6_2'))
        fieldconv7_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv6_2, 16, 16, 3, 2, 'fieldconv7_1'), training=True,name='bnfield7_1'))
        fieldglobalpool = tf.reduce_mean(fieldconv7_1, [2, 3], keep_dims=True)
        fieldflatten = tf.reshape(fieldglobalpool, (1, 64))
        fielddense = relu(tf.layers.dense(fieldflatten, 256))
        fielddense1 = tf.sigmoid(tf.layers.dense(fielddense, 256))
        fieldweight = (tf.reshape(fielddense1, (1, 16, 1, 1, 16)))
        fieldin = fieldconv5_2 * fieldweight




        encodeinput=tf.concat([magin,fieldin,tkdin],4)
        encode1_1=relu(tf.layers.batch_normalization(conv3d(encodeinput,48,32,3,1,'encode1_1'), training=True, name='bnen1'))
        encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=True, name='bnen1_2'))
        encode1_3 = relu(tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 1, 'encode1_3')+encode1_1, training=True, name='bnen1_3'))

        encode2_1=max_pool3d(encode1_3)
        encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=True, name='bnen2_2'))
        encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=True, name='bnen2_3'))
        encode2_4 = relu(tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 1, 'encode2_4')+encode2_2, training=True, name='bnen2_4'))

        encode3_1=max_pool3d(encode2_4)
        encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=True, name='bnen3_2'))
        encode3_3 = relu(tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=True, name='bnen3_3'))
        encode3_4 = relu(tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 1, 'encode3_4')+encode3_2, training=True, name='bnen3_4'))

        encode4_1 = max_pool3d(encode3_4)
        encode4_3 = relu(tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_3'), training=True, name='bnen4_3'))
        encode4_4 = relu(tf.layers.batch_normalization(conv3d(encode4_3, 256, 256, 3, 1, 'encode4_4'), training=True, name='bnen4_4'))
        encode4_5 = relu(tf.layers.batch_normalization(conv3d(encode4_4, 256, 256, 3, 1, 'encode4_5')+encode4_3, training=True, name='bnen4_5'))

        decode3_1 = relu(tf.layers.batch_normalization(deconv3d(encode4_5, 256, 128, 3, 2, 'decode3_1'), training=True,
                                                       name='bnde3_1'))
        decode3_2=tf.concat([decode3_1,encode3_4],4)
        decode3_4 = relu(tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=True, name='bnde3_4'))
        decode3_5 = relu(tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_5'), training=True, name='bnde3_5'))
        decode3_6 = relu(tf.layers.batch_normalization(conv3d(decode3_5, 128, 128, 3, 1, 'decode3_6')+decode3_4, training=True, name='bnde3_6'))

        decode4_1 = relu(tf.layers.batch_normalization(deconv3d(decode3_6, 128, 64, 3, 2, 'decode2_1'), training=True,
                                          name='bnde2_1'))
        decode4_2 = tf.concat([decode4_1,encode2_4],4)
        decode4_4 = relu(tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=True, name='bnde4_4'))
        decode4_5 = relu(tf.layers.batch_normalization(conv3d(decode4_4, 64, 64, 3, 1, 'decode4_5'), training=True, name='bnde4_5'))
        decode4_6 = relu(tf.layers.batch_normalization(conv3d(decode4_5, 64, 64, 3, 1, 'decode4_6')+decode4_4, training=True, name='bnde4_6'))

        decode5_1 = relu(tf.layers.batch_normalization(deconv3d(decode4_6, 64, 32, 3, 2, 'decode1_1'), training=True,
                                                       name='bnde1_1'))
        decode5_2 = tf.concat([decode5_1,encode1_3],4)
        decode5_4 = relu(tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=True, name='bnde5_4'))
        decode5_5 = relu(tf.layers.batch_normalization(conv3d(decode5_4, 32, 32, 3, 1, 'decode5_5'), training=True, name='bnde5_5'))
        decode5_6 = relu(tf.layers.batch_normalization(conv3d(decode5_5, 32, 32, 3, 1, 'decode5_6')+decode5_4, training=True, name='bnde5_6'))
        cosout=conv3d_b(decode5_6, 32, 1, 3, 1, 'out')+tkdinput


        return cosout
def synthesis_newarc2(maginput,fieldinput,tkdinput):
    with tf.variable_scope('syn'):
        #input2=tf.concat([maginput,fieldinput,tkdinput],4)
        # input2_112= tf.reshape(input2,[32,32,32])
        # input2_56 = tf.image.resize_images(input2, [16, 16,16])
        # input2_28 = tf.image.resize_images(input2, [8, 8,8])
        magconv1=relu(tf.layers.batch_normalization(conv3d(maginput,1,16,5,1,'magconv1'), training=True, name='bnmag1'))
        magconv2_1 = relu(tf.layers.batch_normalization(conv3d(magconv1, 16, 16, 3, 1, 'magconv2_1'), training=True, name='bnmag2_1'))
        magconv2_2 = relu(tf.layers.batch_normalization(conv3d(magconv2_1, 16, 16, 3, 1, 'magconv2_2')+magconv1, training=True, name='bnmag2_2'))
        magconv3_1 = relu(tf.layers.batch_normalization(conv3d(magconv2_2, 16, 16, 3, 1, 'magconv3_1'), training=True, name='bnmag3_1'))
        magconv3_2 = relu(tf.layers.batch_normalization(conv3d(magconv3_1, 16, 16, 3, 1, 'magconv3_2') + magconv2_2, training=True, name='bnmag3_2'))
        magconv4_1 = relu(tf.layers.batch_normalization(conv3d(magconv3_2, 16, 16, 3, 1, 'magconv4_1'), training=True, name='bnmag4_1'))
        magconv4_2 = relu(tf.layers.batch_normalization(conv3d(magconv4_1, 16, 16, 3, 1, 'magconv4_2') + magconv3_2, training=True, name='bnmag4_2'))
        magconv5_1 = relu(tf.layers.batch_normalization(conv3d(magconv4_2, 16, 16, 3, 1, 'magconv5_1'), training=True, name='bnmag5_1'))
        magconv5_2 = relu(tf.layers.batch_normalization(conv3d(magconv5_1, 16, 16, 3, 1, 'magconv5_2') + magconv4_2, training=True, name='bnmag5_2'))
        magconv6_1 = relu(tf.layers.batch_normalization(conv3d(magconv5_2, 16, 16, 3, 2, 'magconv6_1'), training=True,name='bnmag6_1'))
        magconv6_2 = relu(tf.layers.batch_normalization(conv3d(magconv6_1, 16, 16, 3, 1, 'magconv6_2'), training=True,name='bnmag6_2'))
        magconv7_1 = relu(tf.layers.batch_normalization(conv3d(magconv6_2, 16, 16, 3, 2, 'magconv7_1'), training=True,name='bnmag7_1'))



        tkdconv1 = relu(
            tf.layers.batch_normalization(conv3d(tkdinput, 1, 16, 5, 1, 'tkdconv1'), training=True, name='bntkd1'))
        tkdconv2_1 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv1, 16, 16, 3, 1, 'tkdconv2_1'), training=True, name='bntkd2_1'))
        tkdconv2_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv2_1, 16, 16, 3, 1, 'tkdconv2_2') + tkdconv1, training=True,
                                          name='bntkd2_2'))
        tkdconv3_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv2_2, 16, 16, 3, 1, 'tkdconv3_1'), training=True,
                                                        name='bntkd3_1'))
        tkdconv3_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv3_1, 16, 16, 3, 1, 'tkdconv3_2') + tkdconv2_2, training=True,
                                          name='bntkd3_2'))
        tkdconv4_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv3_2, 16, 16, 3, 1, 'tkdconv4_1'), training=True,
                                                        name='bntkd4_1'))
        tkdconv4_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv4_1, 16, 16, 3, 1, 'tkdconv4_2') + tkdconv3_2, training=True,
                                          name='bntkd4_2'))
        tkdconv5_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv4_2, 16, 16, 3, 1, 'tkdonv5_1'), training=True,
                                                        name='bntkd5_1'))
        tkdconv5_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv5_1, 16, 16, 3, 1, 'tkdconv5_2') + tkdconv4_2, training=True,
                                          name='bntkd5_2'))
        tkdconv6_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv5_2, 16, 16, 3, 2, 'tkdconv6_1'), training=True,name='bntkd6_1'))
        tkdconv6_2 = relu(tf.layers.batch_normalization(conv3d(tkdconv6_1, 16, 16, 3, 1, 'tkdconv6_2'), training=True,name='bntkd6_2'))
        tkdconv7_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv6_2, 16, 16, 3, 2, 'tkdconv7_1'), training=True,name='bntkd7_1'))



        fieldconv1=relu(tf.layers.batch_normalization(conv3d(fieldinput,1,16,5,1,'fieldconv1'), training=True, name='bnfield1'))
        fieldconv2_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv1, 16, 16, 3, 1, 'fieldconv2_1'), training=True, name='bndield2_1'))
        fieldconv2_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_1, 16, 16, 3, 1, 'fieldconv2_2')+fieldconv1, training=True, name='bnfield2_2'))
        fieldconv3_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_2, 16, 16, 3, 1, 'fieldconv3_1'), training=True, name='bnfield3_1'))
        fieldconv3_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_1, 16, 16, 3, 1, 'fieldconv3_2') + fieldconv2_2, training=True, name='bnfield3_2'))
        fieldconv4_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_2, 16, 16, 3, 1, 'fieldconv4_1'), training=True, name='bnfield4_1'))
        fieldconv4_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_1, 16, 16, 3, 1, 'fieldconv4_2') + fieldconv3_2, training=True, name='bnfield4_2'))
        fieldconv5_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_2, 16, 16, 3, 1, 'fieldconv5_1'), training=True, name='bnfield5_1'))
        fieldconv5_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv5_1, 16, 16, 3, 1, 'fieldconv5_2') + fieldconv4_2, training=True, name='bnfield5_2'))
        fieldconv6_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv5_2, 16, 16, 3, 2, 'fieldconv6_1'), training=True,name='bnfield6_1'))
        fieldconv6_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv6_1, 16, 16, 3, 1, 'fieldconv6_2'), training=True,name='bnfield6_2'))
        fieldconv7_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv6_2, 16, 16, 3, 2, 'fieldconv7_1'), training=True,name='bnfield7_1'))



        concat7_1=tf.concat([magconv7_1,tkdconv7_1,fieldconv7_1],4)
        globalpool=tf.reduce_mean(concat7_1,[2,3],keep_dims=True)
        fusionflatten = tf.reshape(globalpool, (4, 192))
        magdense = tf.layers.dense(fusionflatten, 256)
        tkddense = tf.layers.dense(fusionflatten, 256)
        fielddense = tf.layers.dense(fusionflatten, 256)

        magweight = (tf.reshape(magdense, (4, 16, 1, 1, 16)))
        magin = magconv5_2 * magweight
        tkdweight = (tf.reshape(tkddense, (4, 16, 1, 1, 16)))
        tkdin = tkdconv5_2 * tkdweight
        fieldweight = (tf.reshape(fielddense, (4, 16, 1, 1, 16)))
        fieldin = fieldconv5_2 * fieldweight

        encodeinput=tf.concat([magin,fieldin,tkdin],4)
        encode1_1=relu(tf.layers.batch_normalization(conv3d(encodeinput,48,32,3,1,'encode1_1'), training=True, name='bnen1'))
        encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=True, name='bnen1_2'))
        encode1_3 = relu(tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 1, 'encode1_3')+encode1_1, training=True, name='bnen1_3'))

        encode2_1=max_pool3d(encode1_3)
        encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=True, name='bnen2_2'))
        encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=True, name='bnen2_3'))
        encode2_4 = relu(tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 1, 'encode2_4')+encode2_2, training=True, name='bnen2_4'))

        encode3_1=max_pool3d(encode2_4)
        encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=True, name='bnen3_2'))
        encode3_3 = relu(tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=True, name='bnen3_3'))
        encode3_4 = relu(tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 1, 'encode3_4')+encode3_2, training=True, name='bnen3_4'))

        encode4_1 = max_pool3d(encode3_4)
        encode4_3 = relu(tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_3'), training=True, name='bnen4_3'))
        encode4_4 = relu(tf.layers.batch_normalization(conv3d(encode4_3, 256, 256, 3, 1, 'encode4_4'), training=True, name='bnen4_4'))
        encode4_5 = relu(tf.layers.batch_normalization(conv3d(encode4_4, 256, 256, 3, 1, 'encode4_5')+encode4_3, training=True, name='bnen4_5'))

        decode3_1 = relu(tf.layers.batch_normalization(deconv3d(encode4_5, 256, 128, 3, 2, 'decode3_1'), training=True,
                                                       name='bnde3_1'))
        decode3_2=tf.concat([decode3_1,encode3_4],4)
        decode3_4 = relu(tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=True, name='bnde3_4'))
        decode3_5 = relu(tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_5'), training=True, name='bnde3_5'))
        decode3_6 = relu(tf.layers.batch_normalization(conv3d(decode3_5, 128, 128, 3, 1, 'decode3_6')+decode3_4, training=True, name='bnde3_6'))

        decode4_1 = relu(tf.layers.batch_normalization(deconv3d(decode3_6, 128, 64, 3, 2, 'decode2_1'), training=True,
                                          name='bnde2_1'))
        decode4_2 = tf.concat([decode4_1,encode2_4],4)
        decode4_4 = relu(tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=True, name='bnde4_4'))
        decode4_5 = relu(tf.layers.batch_normalization(conv3d(decode4_4, 64, 64, 3, 1, 'decode4_5'), training=True, name='bnde4_5'))
        decode4_6 = relu(tf.layers.batch_normalization(conv3d(decode4_5, 64, 64, 3, 1, 'decode4_6')+decode4_4, training=True, name='bnde4_6'))

        decode5_1 = relu(tf.layers.batch_normalization(deconv3d(decode4_6, 64, 32, 3, 2, 'decode1_1'), training=True,
                                                       name='bnde1_1'))
        decode5_2 = tf.concat([decode5_1,encode1_3],4)
        decode5_4 = relu(tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=True, name='bnde5_4'))
        decode5_5 = relu(tf.layers.batch_normalization(conv3d(decode5_4, 32, 32, 3, 1, 'decode5_5'), training=True, name='bnde5_5'))
        decode5_6 = relu(tf.layers.batch_normalization(conv3d(decode5_5, 32, 32, 3, 1, 'decode5_6')+decode5_4, training=True, name='bnde5_6'))
        cosout=conv3d_b(decode5_6, 32, 1, 3, 1, 'out')+tkdinput

    with tf.variable_scope('ref'):
        fea = Feature_Extraction(cosout)
        refencodeinput = tf.concat([cosout, fea], 4)
        refencode1_1 = relu(
            tf.layers.batch_normalization(conv3d(refencodeinput, 10, 32, 3, 1, 'encode1_1'), training=True,
                                          name='bnen1'))
        refencode1_2 = relu(
            tf.layers.batch_normalization(conv3d(refencode1_1, 32, 32, 3, 1, 'encode1_2'), training=True,
                                          name='bnen1_2'))
        refencode1_3 = relu(
            tf.layers.batch_normalization(conv3d(refencode1_2, 32, 32, 3, 1, 'encode1_3') + refencode1_1, training=True,
                                          name='bnen1_3'))

        refencode2_1 = max_pool3d(refencode1_3)
        refencode2_2 = relu(
            tf.layers.batch_normalization(conv3d(refencode2_1, 32, 64, 3, 1, 'encode2_2'), training=True,
                                          name='bnen2_2'))
        refencode2_3 = relu(
            tf.layers.batch_normalization(conv3d(refencode2_2, 64, 64, 3, 1, 'encode2_3'), training=True,
                                          name='bnen2_3'))
        refencode2_4 = relu(
            tf.layers.batch_normalization(conv3d(refencode2_3, 64, 64, 3, 1, 'encode2_4') + refencode2_2, training=True,
                                          name='bnen2_4'))

        refencode3_1 = max_pool3d(refencode2_4)
        refencode3_2 = relu(
            tf.layers.batch_normalization(conv3d(refencode3_1, 64, 128, 3, 1, 'encode3_2'), training=True,
                                          name='bnen3_2'))
        refencode3_3 = relu(
            tf.layers.batch_normalization(conv3d(refencode3_2, 128, 128, 3, 1, 'encode3_3'), training=True,
                                          name='bnen3_3'))
        refencode3_4 = relu(
            tf.layers.batch_normalization(conv3d(refencode3_3, 128, 128, 3, 1, 'encode3_4') + refencode3_2,
                                          training=True,
                                          name='bnen3_4'))

        refencode4_1 = max_pool3d(refencode3_4)
        refencode4_3 = relu(
            tf.layers.batch_normalization(conv3d(refencode4_1, 128, 256, 3, 1, 'encode4_3'), training=True,
                                          name='bnen4_3'))
        refencode4_4 = relu(
            tf.layers.batch_normalization(conv3d(refencode4_3, 256, 256, 3, 1, 'encode4_4'), training=True,
                                          name='bnen4_4'))
        refencode4_5 = relu(
            tf.layers.batch_normalization(conv3d(refencode4_4, 256, 256, 3, 1, 'encode4_5') + refencode4_3,
                                          training=True,
                                          name='bnen4_5'))

        refdecode3_1 = relu(
            tf.layers.batch_normalization(deconv3d(refencode4_5, 256, 128, 3, 2, 'decode3_1'), training=True,
                                          name='bnde3_1'))
        refdecode3_2 = tf.concat([refdecode3_1, refencode3_4], 4)
        refdecode3_4 = relu(
            tf.layers.batch_normalization(conv3d(refdecode3_2, 256, 128, 3, 1, 'decode3_4'), training=True,
                                          name='bnde3_4'))
        refdecode3_5 = relu(
            tf.layers.batch_normalization(conv3d(refdecode3_4, 128, 128, 3, 1, 'decode3_5'), training=True,
                                          name='bnde3_5'))
        refdecode3_6 = relu(
            tf.layers.batch_normalization(conv3d(refdecode3_5, 128, 128, 3, 1, 'decode3_6') + refdecode3_4,
                                          training=True,
                                          name='bnde3_6'))

        refdecode4_1 = relu(
            tf.layers.batch_normalization(deconv3d(refdecode3_6, 128, 64, 3, 2, 'decode2_1'), training=True,
                                          name='bnde2_1'))
        refdecode4_2 = tf.concat([refdecode4_1, refencode2_4], 4)
        refdecode4_4 = relu(
            tf.layers.batch_normalization(conv3d(refdecode4_2, 128, 64, 3, 1, 'decode4_4'), training=True,
                                          name='bnde4_4'))
        refdecode4_5 = relu(
            tf.layers.batch_normalization(conv3d(refdecode4_4, 64, 64, 3, 1, 'decode4_5'), training=True,
                                          name='bnde4_5'))
        refdecode4_6 = relu(
            tf.layers.batch_normalization(conv3d(refdecode4_5, 64, 64, 3, 1, 'decode4_6') + refdecode4_4, training=True,
                                          name='bnde4_6'))

        refdecode5_1 = relu(
            tf.layers.batch_normalization(deconv3d(refdecode4_6, 64, 32, 3, 2, 'decode1_1'), training=True,
                                          name='bnde1_1'))
        refdecode5_2 = tf.concat([refdecode5_1, refencode1_3], 4)
        refdecode5_4 = relu(
            tf.layers.batch_normalization(conv3d(refdecode5_2, 64, 32, 3, 1, 'decode5_4'), training=True,
                                          name='bnde5_4'))
        refdecode5_5 = relu(
            tf.layers.batch_normalization(conv3d(refdecode5_4, 32, 32, 3, 1, 'decode5_5'), training=True,
                                          name='bnde5_5'))
        refdecode5_6 = relu(
            tf.layers.batch_normalization(conv3d(refdecode5_5, 32, 32, 3, 1, 'decode5_6') + refdecode5_4, training=True,
                                          name='bnde5_6'))
        refcosout = conv3d_b(refdecode5_6, 32, 1, 3, 1, 'out') + cosout
    return cosout, refcosout
def t_synthesis_newarc2(maginput,fieldinput,tkdinput):
    with tf.variable_scope('syn'):
        #input2=tf.concat([maginput,fieldinput,tkdinput],4)
        # input2_112= tf.reshape(input2,[32,32,32])
        # input2_56 = tf.image.resize_images(input2, [16, 16,16])
        # input2_28 = tf.image.resize_images(input2, [8, 8,8])
        magconv1=relu(tf.layers.batch_normalization(conv3d(maginput,1,16,5,1,'magconv1'), training=True, name='bnmag1'))
        magconv2_1 = relu(tf.layers.batch_normalization(conv3d(magconv1, 16, 16, 3, 1, 'magconv2_1'), training=True, name='bnmag2_1'))
        magconv2_2 = relu(tf.layers.batch_normalization(conv3d(magconv2_1, 16, 16, 3, 1, 'magconv2_2')+magconv1, training=True, name='bnmag2_2'))
        magconv3_1 = relu(tf.layers.batch_normalization(conv3d(magconv2_2, 16, 16, 3, 1, 'magconv3_1'), training=True, name='bnmag3_1'))
        magconv3_2 = relu(tf.layers.batch_normalization(conv3d(magconv3_1, 16, 16, 3, 1, 'magconv3_2') + magconv2_2, training=True, name='bnmag3_2'))
        magconv4_1 = relu(tf.layers.batch_normalization(conv3d(magconv3_2, 16, 16, 3, 1, 'magconv4_1'), training=True, name='bnmag4_1'))
        magconv4_2 = relu(tf.layers.batch_normalization(conv3d(magconv4_1, 16, 16, 3, 1, 'magconv4_2') + magconv3_2, training=True, name='bnmag4_2'))
        magconv5_1 = relu(tf.layers.batch_normalization(conv3d(magconv4_2, 16, 16, 3, 1, 'magconv5_1'), training=True, name='bnmag5_1'))
        magconv5_2 = relu(tf.layers.batch_normalization(conv3d(magconv5_1, 16, 16, 3, 1, 'magconv5_2') + magconv4_2, training=True, name='bnmag5_2'))
        magconv6_1 = relu(tf.layers.batch_normalization(conv3d(magconv5_2, 16, 16, 3, 2, 'magconv6_1'), training=True,name='bnmag6_1'))
        magconv6_2 = relu(tf.layers.batch_normalization(conv3d(magconv6_1, 16, 16, 3, 1, 'magconv6_2'), training=True,name='bnmag6_2'))
        magconv7_1 = relu(tf.layers.batch_normalization(conv3d(magconv6_2, 16, 16, 3, 2, 'magconv7_1'), training=True,name='bnmag7_1'))



        tkdconv1 = relu(
            tf.layers.batch_normalization(conv3d(tkdinput, 1, 16, 5, 1, 'tkdconv1'), training=True, name='bntkd1'))
        tkdconv2_1 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv1, 16, 16, 3, 1, 'tkdconv2_1'), training=True, name='bntkd2_1'))
        tkdconv2_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv2_1, 16, 16, 3, 1, 'tkdconv2_2') + tkdconv1, training=True,
                                          name='bntkd2_2'))
        tkdconv3_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv2_2, 16, 16, 3, 1, 'tkdconv3_1'), training=True,
                                                        name='bntkd3_1'))
        tkdconv3_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv3_1, 16, 16, 3, 1, 'tkdconv3_2') + tkdconv2_2, training=True,
                                          name='bntkd3_2'))
        tkdconv4_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv3_2, 16, 16, 3, 1, 'tkdconv4_1'), training=True,
                                                        name='bntkd4_1'))
        tkdconv4_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv4_1, 16, 16, 3, 1, 'tkdconv4_2') + tkdconv3_2, training=True,
                                          name='bntkd4_2'))
        tkdconv5_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv4_2, 16, 16, 3, 1, 'tkdonv5_1'), training=True,
                                                        name='bntkd5_1'))
        tkdconv5_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv5_1, 16, 16, 3, 1, 'tkdconv5_2') + tkdconv4_2, training=True,
                                          name='bntkd5_2'))
        tkdconv6_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv5_2, 16, 16, 3, 2, 'tkdconv6_1'), training=True,name='bntkd6_1'))
        tkdconv6_2 = relu(tf.layers.batch_normalization(conv3d(tkdconv6_1, 16, 16, 3, 1, 'tkdconv6_2'), training=True,name='bntkd6_2'))
        tkdconv7_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv6_2, 16, 16, 3, 2, 'tkdconv7_1'), training=True,name='bntkd7_1'))



        fieldconv1=relu(tf.layers.batch_normalization(conv3d(fieldinput,1,16,5,1,'fieldconv1'), training=True, name='bnfield1'))
        fieldconv2_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv1, 16, 16, 3, 1, 'fieldconv2_1'), training=True, name='bndield2_1'))
        fieldconv2_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_1, 16, 16, 3, 1, 'fieldconv2_2')+fieldconv1, training=True, name='bnfield2_2'))
        fieldconv3_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_2, 16, 16, 3, 1, 'fieldconv3_1'), training=True, name='bnfield3_1'))
        fieldconv3_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_1, 16, 16, 3, 1, 'fieldconv3_2') + fieldconv2_2, training=True, name='bnfield3_2'))
        fieldconv4_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_2, 16, 16, 3, 1, 'fieldconv4_1'), training=True, name='bnfield4_1'))
        fieldconv4_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_1, 16, 16, 3, 1, 'fieldconv4_2') + fieldconv3_2, training=True, name='bnfield4_2'))
        fieldconv5_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_2, 16, 16, 3, 1, 'fieldconv5_1'), training=True, name='bnfield5_1'))
        fieldconv5_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv5_1, 16, 16, 3, 1, 'fieldconv5_2') + fieldconv4_2, training=True, name='bnfield5_2'))
        fieldconv6_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv5_2, 16, 16, 3, 2, 'fieldconv6_1'), training=True,name='bnfield6_1'))
        fieldconv6_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv6_1, 16, 16, 3, 1, 'fieldconv6_2'), training=True,name='bnfield6_2'))
        fieldconv7_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv6_2, 16, 16, 3, 2, 'fieldconv7_1'), training=True,name='bnfield7_1'))



        concat7_1=tf.concat([magconv7_1,tkdconv7_1,fieldconv7_1],4)
        globalpool=tf.reduce_mean(concat7_1,[2,3],keep_dims=True)
        fusionflatten = tf.reshape(globalpool, (1, 192))
        magdense = tf.layers.dense(fusionflatten, 256)
        tkddense = tf.layers.dense(fusionflatten, 256)
        fielddense = tf.layers.dense(fusionflatten, 256)

        magweight = (tf.reshape(magdense, (1, 16, 1, 1, 16)))
        magin = magconv5_2 * magweight
        tkdweight = (tf.reshape(tkddense, (1, 16, 1, 1, 16)))
        tkdin = tkdconv5_2 * tkdweight
        fieldweight = (tf.reshape(fielddense, (1, 16, 1, 1, 16)))
        fieldin = fieldconv5_2 * fieldweight

        encodeinput=tf.concat([magin,fieldin,tkdin],4)
        encode1_1=relu(tf.layers.batch_normalization(conv3d(encodeinput,48,32,3,1,'encode1_1'), training=True, name='bnen1'))
        encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=True, name='bnen1_2'))
        encode1_3 = relu(tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 1, 'encode1_3')+encode1_1, training=True, name='bnen1_3'))

        encode2_1=max_pool3d(encode1_3)
        encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=True, name='bnen2_2'))
        encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=True, name='bnen2_3'))
        encode2_4 = relu(tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 1, 'encode2_4')+encode2_2, training=True, name='bnen2_4'))

        encode3_1=max_pool3d(encode2_4)
        encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=True, name='bnen3_2'))
        encode3_3 = relu(tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=True, name='bnen3_3'))
        encode3_4 = relu(tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 1, 'encode3_4')+encode3_2, training=True, name='bnen3_4'))

        encode4_1 = max_pool3d(encode3_4)
        encode4_3 = relu(tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_3'), training=True, name='bnen4_3'))
        encode4_4 = relu(tf.layers.batch_normalization(conv3d(encode4_3, 256, 256, 3, 1, 'encode4_4'), training=True, name='bnen4_4'))
        encode4_5 = relu(tf.layers.batch_normalization(conv3d(encode4_4, 256, 256, 3, 1, 'encode4_5')+encode4_3, training=True, name='bnen4_5'))

        decode3_1 = relu(tf.layers.batch_normalization(deconv3d(encode4_5, 256, 128, 3, 2, 'decode3_1'), training=True,
                                                       name='bnde3_1'))
        decode3_2=tf.concat([decode3_1,encode3_4],4)
        decode3_4 = relu(tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=True, name='bnde3_4'))
        decode3_5 = relu(tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_5'), training=True, name='bnde3_5'))
        decode3_6 = relu(tf.layers.batch_normalization(conv3d(decode3_5, 128, 128, 3, 1, 'decode3_6')+decode3_4, training=True, name='bnde3_6'))

        decode4_1 = relu(tf.layers.batch_normalization(deconv3d(decode3_6, 128, 64, 3, 2, 'decode2_1'), training=True,
                                          name='bnde2_1'))
        decode4_2 = tf.concat([decode4_1,encode2_4],4)
        decode4_4 = relu(tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=True, name='bnde4_4'))
        decode4_5 = relu(tf.layers.batch_normalization(conv3d(decode4_4, 64, 64, 3, 1, 'decode4_5'), training=True, name='bnde4_5'))
        decode4_6 = relu(tf.layers.batch_normalization(conv3d(decode4_5, 64, 64, 3, 1, 'decode4_6')+decode4_4, training=True, name='bnde4_6'))

        decode5_1 = relu(tf.layers.batch_normalization(deconv3d(decode4_6, 64, 32, 3, 2, 'decode1_1'), training=True,
                                                       name='bnde1_1'))
        decode5_2 = tf.concat([decode5_1,encode1_3],4)
        decode5_4 = relu(tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=True, name='bnde5_4'))
        decode5_5 = relu(tf.layers.batch_normalization(conv3d(decode5_4, 32, 32, 3, 1, 'decode5_5'), training=True, name='bnde5_5'))
        decode5_6 = relu(tf.layers.batch_normalization(conv3d(decode5_5, 32, 32, 3, 1, 'decode5_6')+decode5_4, training=True, name='bnde5_6'))
        cosout=conv3d_b(decode5_6, 32, 1, 3, 1, 'out')+tkdinput

    with tf.variable_scope('ref'):
        fea = Feature_Extraction(cosout)
        refencodeinput = tf.concat([cosout, fea], 4)
        refencode1_1 = relu(
            tf.layers.batch_normalization(conv3d(refencodeinput, 10, 32, 3, 1, 'encode1_1'), training=True,
                                          name='bnen1'))
        refencode1_2 = relu(
            tf.layers.batch_normalization(conv3d(refencode1_1, 32, 32, 3, 1, 'encode1_2'), training=True,
                                          name='bnen1_2'))
        refencode1_3 = relu(
            tf.layers.batch_normalization(conv3d(refencode1_2, 32, 32, 3, 1, 'encode1_3') + refencode1_1, training=True,
                                          name='bnen1_3'))

        refencode2_1 = max_pool3d(refencode1_3)
        refencode2_2 = relu(
            tf.layers.batch_normalization(conv3d(refencode2_1, 32, 64, 3, 1, 'encode2_2'), training=True,
                                          name='bnen2_2'))
        refencode2_3 = relu(
            tf.layers.batch_normalization(conv3d(refencode2_2, 64, 64, 3, 1, 'encode2_3'), training=True,
                                          name='bnen2_3'))
        refencode2_4 = relu(
            tf.layers.batch_normalization(conv3d(refencode2_3, 64, 64, 3, 1, 'encode2_4') + refencode2_2, training=True,
                                          name='bnen2_4'))

        refencode3_1 = max_pool3d(refencode2_4)
        refencode3_2 = relu(
            tf.layers.batch_normalization(conv3d(refencode3_1, 64, 128, 3, 1, 'encode3_2'), training=True,
                                          name='bnen3_2'))
        refencode3_3 = relu(
            tf.layers.batch_normalization(conv3d(refencode3_2, 128, 128, 3, 1, 'encode3_3'), training=True,
                                          name='bnen3_3'))
        refencode3_4 = relu(
            tf.layers.batch_normalization(conv3d(refencode3_3, 128, 128, 3, 1, 'encode3_4') + refencode3_2,
                                          training=True,
                                          name='bnen3_4'))

        refencode4_1 = max_pool3d(refencode3_4)
        refencode4_3 = relu(
            tf.layers.batch_normalization(conv3d(refencode4_1, 128, 256, 3, 1, 'encode4_3'), training=True,
                                          name='bnen4_3'))
        refencode4_4 = relu(
            tf.layers.batch_normalization(conv3d(refencode4_3, 256, 256, 3, 1, 'encode4_4'), training=True,
                                          name='bnen4_4'))
        refencode4_5 = relu(
            tf.layers.batch_normalization(conv3d(refencode4_4, 256, 256, 3, 1, 'encode4_5') + refencode4_3,
                                          training=True,
                                          name='bnen4_5'))

        refdecode3_1 = relu(
            tf.layers.batch_normalization(deconv3d(refencode4_5, 256, 128, 3, 2, 'decode3_1'), training=True,
                                          name='bnde3_1'))
        refdecode3_2 = tf.concat([refdecode3_1, refencode3_4], 4)
        refdecode3_4 = relu(
            tf.layers.batch_normalization(conv3d(refdecode3_2, 256, 128, 3, 1, 'decode3_4'), training=True,
                                          name='bnde3_4'))
        refdecode3_5 = relu(
            tf.layers.batch_normalization(conv3d(refdecode3_4, 128, 128, 3, 1, 'decode3_5'), training=True,
                                          name='bnde3_5'))
        refdecode3_6 = relu(
            tf.layers.batch_normalization(conv3d(refdecode3_5, 128, 128, 3, 1, 'decode3_6') + refdecode3_4,
                                          training=True,
                                          name='bnde3_6'))

        refdecode4_1 = relu(
            tf.layers.batch_normalization(deconv3d(refdecode3_6, 128, 64, 3, 2, 'decode2_1'), training=True,
                                          name='bnde2_1'))
        refdecode4_2 = tf.concat([refdecode4_1, refencode2_4], 4)
        refdecode4_4 = relu(
            tf.layers.batch_normalization(conv3d(refdecode4_2, 128, 64, 3, 1, 'decode4_4'), training=True,
                                          name='bnde4_4'))
        refdecode4_5 = relu(
            tf.layers.batch_normalization(conv3d(refdecode4_4, 64, 64, 3, 1, 'decode4_5'), training=True,
                                          name='bnde4_5'))
        refdecode4_6 = relu(
            tf.layers.batch_normalization(conv3d(refdecode4_5, 64, 64, 3, 1, 'decode4_6') + refdecode4_4, training=True,
                                          name='bnde4_6'))

        refdecode5_1 = relu(
            tf.layers.batch_normalization(deconv3d(refdecode4_6, 64, 32, 3, 2, 'decode1_1'), training=True,
                                          name='bnde1_1'))
        refdecode5_2 = tf.concat([refdecode5_1, refencode1_3], 4)
        refdecode5_4 = relu(
            tf.layers.batch_normalization(conv3d(refdecode5_2, 64, 32, 3, 1, 'decode5_4'), training=True,
                                          name='bnde5_4'))
        refdecode5_5 = relu(
            tf.layers.batch_normalization(conv3d(refdecode5_4, 32, 32, 3, 1, 'decode5_5'), training=True,
                                          name='bnde5_5'))
        refdecode5_6 = relu(
            tf.layers.batch_normalization(conv3d(refdecode5_5, 32, 32, 3, 1, 'decode5_6') + refdecode5_4, training=True,
                                          name='bnde5_6'))
        refcosout = conv3d_b(refdecode5_6, 32, 1, 3, 1, 'out') + cosout
    return cosout, refcosout
def synthesis_newarc_ref(maginput,fieldinput,tkdinput):
    with tf.variable_scope('syn'):
        with tf.variable_scope('rec'):
            #input2=tf.concat([maginput,fieldinput,tkdinput],4)
            # input2_112= tf.reshape(input2,[32,32,32])
            # input2_56 = tf.image.resize_images(input2, [16, 16,16])
            # input2_28 = tf.image.resize_images(input2, [8, 8,8])
            magconv1=relu(tf.layers.batch_normalization(conv3d(maginput,1,16,5,1,'magconv1'), training=True, name='bnmag1'))
            magconv2_1 = relu(tf.layers.batch_normalization(conv3d(magconv1, 16, 16, 3, 1, 'magconv2_1'), training=True, name='bnmag2_1'))
            magconv2_2 = relu(tf.layers.batch_normalization(conv3d(magconv2_1, 16, 16, 3, 1, 'magconv2_2')+magconv1, training=True, name='bnmag2_2'))
            magconv3_1 = relu(tf.layers.batch_normalization(conv3d(magconv2_2, 16, 16, 3, 1, 'magconv3_1'), training=True, name='bnmag3_1'))
            magconv3_2 = relu(tf.layers.batch_normalization(conv3d(magconv3_1, 16, 16, 3, 1, 'magconv3_2') + magconv2_2, training=True, name='bnmag3_2'))
            magconv4_1 = relu(tf.layers.batch_normalization(conv3d(magconv3_2, 16, 16, 3, 1, 'magconv4_1'), training=True, name='bnmag4_1'))
            magconv4_2 = relu(tf.layers.batch_normalization(conv3d(magconv4_1, 16, 16, 3, 1, 'magconv4_2') + magconv3_2, training=True, name='bnmag4_2'))
            magconv5_1 = relu(tf.layers.batch_normalization(conv3d(magconv4_2, 16, 16, 3, 1, 'magconv5_1'), training=True, name='bnmag5_1'))
            magconv5_2 = relu(tf.layers.batch_normalization(conv3d(magconv5_1, 16, 16, 3, 1, 'magconv5_2') + magconv4_2, training=True, name='bnmag5_2'))
            magconv6_1 = relu(tf.layers.batch_normalization(conv3d(magconv5_2, 16, 16, 3, 2, 'magconv6_1'), training=True,name='bnmag6_1'))
            magconv6_2 = relu(tf.layers.batch_normalization(conv3d(magconv6_1, 16, 16, 3, 1, 'magconv6_2'), training=True,name='bnmag6_2'))
            magconv7_1 = relu(tf.layers.batch_normalization(conv3d(magconv6_2, 16, 16, 3, 2, 'magconv7_1'), training=True,name='bnmag7_1'))



            tkdconv1 = relu(
                tf.layers.batch_normalization(conv3d(tkdinput, 1, 16, 5, 1, 'tkdconv1'), training=True, name='bntkd1'))
            tkdconv2_1 = relu(
                tf.layers.batch_normalization(conv3d(tkdconv1, 16, 16, 3, 1, 'tkdconv2_1'), training=True, name='bntkd2_1'))
            tkdconv2_2 = relu(
                tf.layers.batch_normalization(conv3d(tkdconv2_1, 16, 16, 3, 1, 'tkdconv2_2') + tkdconv1, training=True,
                                              name='bntkd2_2'))
            tkdconv3_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv2_2, 16, 16, 3, 1, 'tkdconv3_1'), training=True,
                                                            name='bntkd3_1'))
            tkdconv3_2 = relu(
                tf.layers.batch_normalization(conv3d(tkdconv3_1, 16, 16, 3, 1, 'tkdconv3_2') + tkdconv2_2, training=True,
                                              name='bntkd3_2'))
            tkdconv4_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv3_2, 16, 16, 3, 1, 'tkdconv4_1'), training=True,
                                                            name='bntkd4_1'))
            tkdconv4_2 = relu(
                tf.layers.batch_normalization(conv3d(tkdconv4_1, 16, 16, 3, 1, 'tkdconv4_2') + tkdconv3_2, training=True,
                                              name='bntkd4_2'))
            tkdconv5_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv4_2, 16, 16, 3, 1, 'tkdonv5_1'), training=True,
                                                            name='bntkd5_1'))
            tkdconv5_2 = relu(
                tf.layers.batch_normalization(conv3d(tkdconv5_1, 16, 16, 3, 1, 'tkdconv5_2') + tkdconv4_2, training=True,
                                              name='bntkd5_2'))
            tkdconv6_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv5_2, 16, 16, 3, 2, 'tkdconv6_1'), training=True,name='bntkd6_1'))
            tkdconv6_2 = relu(tf.layers.batch_normalization(conv3d(tkdconv6_1, 16, 16, 3, 1, 'tkdconv6_2'), training=True,name='bntkd6_2'))
            tkdconv7_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv6_2, 16, 16, 3, 2, 'tkdconv7_1'), training=True,name='bntkd7_1'))



            fieldconv1=relu(tf.layers.batch_normalization(conv3d(fieldinput,1,16,5,1,'fieldconv1'), training=True, name='bnfield1'))
            fieldconv2_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv1, 16, 16, 3, 1, 'fieldconv2_1'), training=True, name='bndield2_1'))
            fieldconv2_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_1, 16, 16, 3, 1, 'fieldconv2_2')+fieldconv1, training=True, name='bnfield2_2'))
            fieldconv3_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_2, 16, 16, 3, 1, 'fieldconv3_1'), training=True, name='bnfield3_1'))
            fieldconv3_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_1, 16, 16, 3, 1, 'fieldconv3_2') + fieldconv2_2, training=True, name='bnfield3_2'))
            fieldconv4_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_2, 16, 16, 3, 1, 'fieldconv4_1'), training=True, name='bnfield4_1'))
            fieldconv4_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_1, 16, 16, 3, 1, 'fieldconv4_2') + fieldconv3_2, training=True, name='bnfield4_2'))
            fieldconv5_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_2, 16, 16, 3, 1, 'fieldconv5_1'), training=True, name='bnfield5_1'))
            fieldconv5_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv5_1, 16, 16, 3, 1, 'fieldconv5_2') + fieldconv4_2, training=True, name='bnfield5_2'))
            fieldconv6_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv5_2, 16, 16, 3, 2, 'fieldconv6_1'), training=True,name='bnfield6_1'))
            fieldconv6_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv6_1, 16, 16, 3, 1, 'fieldconv6_2'), training=True,name='bnfield6_2'))
            fieldconv7_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv6_2, 16, 16, 3, 2, 'fieldconv7_1'), training=True,name='bnfield7_1'))



            concat7_1=tf.concat([magconv7_1,tkdconv7_1,fieldconv7_1],4)
            globalpool = tf.reduce_mean(concat7_1, [2, 3], keep_dims=True)
            fusionflatten = tf.reshape(globalpool, (4, 192))
            magdense = tf.layers.dense(fusionflatten, 256)
            tkddense = tf.layers.dense(fusionflatten, 256)
            fielddense = tf.layers.dense(fusionflatten, 256)

            magweight = (tf.reshape(magdense, (4, 16, 1, 1, 16)))
            magin = magconv5_2 * magweight
            tkdweight = (tf.reshape(tkddense, (4, 16, 1, 1, 16)))
            tkdin = tkdconv5_2 * tkdweight
            fieldweight = (tf.reshape(fielddense, (4, 16, 1, 1, 16)))
            fieldin = fieldconv5_2 * fieldweight

            encodeinput=tf.concat([magin,fieldin,tkdin],4)
            encode1_1=relu(tf.layers.batch_normalization(conv3d(encodeinput,48,32,3,1,'encode1_1'), training=True, name='bnen1'))
            encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=True, name='bnen1_2'))
            encode1_3 = relu(tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 1, 'encode1_3')+encode1_1, training=True, name='bnen1_3'))

            encode2_1=max_pool3d(encode1_3)
            encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=True, name='bnen2_2'))
            encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=True, name='bnen2_3'))
            encode2_4 = relu(tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 1, 'encode2_4')+encode2_2, training=True, name='bnen2_4'))

            encode3_1=max_pool3d(encode2_4)
            encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=True, name='bnen3_2'))
            encode3_3 = relu(tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=True, name='bnen3_3'))
            encode3_4 = relu(tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 1, 'encode3_4')+encode3_2, training=True, name='bnen3_4'))

            encode4_1 = max_pool3d(encode3_4)
            encode4_3 = relu(tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_3'), training=True, name='bnen4_3'))
            encode4_4 = relu(tf.layers.batch_normalization(conv3d(encode4_3, 256, 256, 3, 1, 'encode4_4'), training=True, name='bnen4_4'))
            encode4_5 = relu(tf.layers.batch_normalization(conv3d(encode4_4, 256, 256, 3, 1, 'encode4_5')+encode4_3, training=True, name='bnen4_5'))

            decode3_1 = relu(tf.layers.batch_normalization(deconv3d(encode4_5, 256, 128, 3, 2, 'decode3_1'), training=True,
                                                           name='bnde3_1'))
            decode3_2=tf.concat([decode3_1,encode3_4], 4)
            decode3_4 = relu(tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=True, name='bnde3_4'))
            decode3_5 = relu(tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_5'), training=True, name='bnde3_5'))
            decode3_6 = relu(tf.layers.batch_normalization(conv3d(decode3_5, 128, 128, 3, 1, 'decode3_6')+decode3_4, training=True, name='bnde3_6'))

            decode4_1 = relu(tf.layers.batch_normalization(deconv3d(decode3_6, 128, 64, 3, 2, 'decode2_1'), training=True,
                                              name='bnde2_1'))
            decode4_2 = tf.concat([decode4_1,encode2_4], 4)
            decode4_4 = relu(tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=True, name='bnde4_4'))
            decode4_5 = relu(tf.layers.batch_normalization(conv3d(decode4_4, 64, 64, 3, 1, 'decode4_5'), training=True, name='bnde4_5'))
            decode4_6 = relu(tf.layers.batch_normalization(conv3d(decode4_5, 64, 64, 3, 1, 'decode4_6')+decode4_4, training=True, name='bnde4_6'))

            decode5_1 = relu(tf.layers.batch_normalization(deconv3d(decode4_6, 64, 32, 3, 2, 'decode1_1'), training=True,
                                                           name='bnde1_1'))
            decode5_2 = tf.concat([decode5_1,encode1_3], 4)
            decode5_4 = relu(tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=True, name='bnde5_4'))
            decode5_5 = relu(tf.layers.batch_normalization(conv3d(decode5_4, 32, 32, 3, 1, 'decode5_5'), training=True, name='bnde5_5'))
            decode5_6 = relu(tf.layers.batch_normalization(conv3d(decode5_5, 32, 32, 3, 1, 'decode5_6')+decode5_4, training=True, name='bnde5_6'))
            cosout=conv3d_b(decode5_6, 32, 1, 3, 1, 'out')+tkdinput
        with tf.variable_scope('ref'):
            fea=Feature_Extraction(cosout)
            refencodeinput = tf.concat([cosout, fea], 4)
            refencode1_1 = relu(
                tf.layers.batch_normalization(conv3d(refencodeinput, 10, 32, 3, 1, 'encode1_1'), training=True, name='bnen1'))
            refencode1_2 = relu(
                tf.layers.batch_normalization(conv3d(refencode1_1, 32, 32, 3, 1, 'encode1_2'), training=True, name='bnen1_2'))
            refencode1_3 = relu(
                tf.layers.batch_normalization(conv3d(refencode1_2, 32, 32, 3, 1, 'encode1_3') + refencode1_1, training=True,
                                              name='bnen1_3'))

            refencode2_1 = max_pool3d(refencode1_3)
            refencode2_2 = relu(
                tf.layers.batch_normalization(conv3d(refencode2_1, 32, 64, 3, 1, 'encode2_2'), training=True, name='bnen2_2'))
            refencode2_3 = relu(
                tf.layers.batch_normalization(conv3d(refencode2_2, 64, 64, 3, 1, 'encode2_3'), training=True, name='bnen2_3'))
            refencode2_4 = relu(
                tf.layers.batch_normalization(conv3d(refencode2_3, 64, 64, 3, 1, 'encode2_4') + refencode2_2, training=True,
                                              name='bnen2_4'))

            refencode3_1 = max_pool3d(refencode2_4)
            refencode3_2 = relu(
                tf.layers.batch_normalization(conv3d(refencode3_1, 64, 128, 3, 1, 'encode3_2'), training=True, name='bnen3_2'))
            refencode3_3 = relu(tf.layers.batch_normalization(conv3d(refencode3_2, 128, 128, 3, 1, 'encode3_3'), training=True,
                                                           name='bnen3_3'))
            refencode3_4 = relu(
                tf.layers.batch_normalization(conv3d(refencode3_3, 128, 128, 3, 1, 'encode3_4') + refencode3_2, training=True,
                                              name='bnen3_4'))

            refencode4_1 = max_pool3d(refencode3_4)
            refencode4_3 = relu(tf.layers.batch_normalization(conv3d(refencode4_1, 128, 256, 3, 1, 'encode4_3'), training=True,
                                                           name='bnen4_3'))
            refencode4_4 = relu(tf.layers.batch_normalization(conv3d(refencode4_3, 256, 256, 3, 1, 'encode4_4'), training=True,
                                                           name='bnen4_4'))
            refencode4_5 = relu(
                tf.layers.batch_normalization(conv3d(refencode4_4, 256, 256, 3, 1, 'encode4_5') + refencode4_3, training=True,
                                              name='bnen4_5'))

            refdecode3_1 = relu(tf.layers.batch_normalization(deconv3d(refencode4_5, 256, 128, 3, 2, 'decode3_1'), training=True,
                                                           name='bnde3_1'))
            refdecode3_2 = tf.concat([refdecode3_1,refencode3_4], 4)
            refdecode3_4 = relu(tf.layers.batch_normalization(conv3d(refdecode3_2, 256, 128, 3, 1, 'decode3_4'), training=True,
                                                           name='bnde3_4'))
            refdecode3_5 = relu(tf.layers.batch_normalization(conv3d(refdecode3_4, 128, 128, 3, 1, 'decode3_5'), training=True,
                                                           name='bnde3_5'))
            refdecode3_6 = relu(
                tf.layers.batch_normalization(conv3d(refdecode3_5, 128, 128, 3, 1, 'decode3_6') + refdecode3_4, training=True,
                                              name='bnde3_6'))

            refdecode4_1 = relu(tf.layers.batch_normalization(deconv3d(refdecode3_6, 128, 64, 3, 2, 'decode2_1'), training=True,
                                                           name='bnde2_1'))
            refdecode4_2 = tf.concat([refdecode4_1,refencode2_4], 4)
            refdecode4_4 = relu(
                tf.layers.batch_normalization(conv3d(refdecode4_2, 128, 64, 3, 1, 'decode4_4'), training=True, name='bnde4_4'))
            refdecode4_5 = relu(
                tf.layers.batch_normalization(conv3d(refdecode4_4, 64, 64, 3, 1, 'decode4_5'), training=True, name='bnde4_5'))
            refdecode4_6 = relu(
                tf.layers.batch_normalization(conv3d(refdecode4_5, 64, 64, 3, 1, 'decode4_6') + refdecode4_4, training=True,
                                              name='bnde4_6'))

            refdecode5_1 = relu(tf.layers.batch_normalization(deconv3d(refdecode4_6, 64, 32, 3, 2, 'decode1_1'), training=True,
                                                           name='bnde1_1'))
            refdecode5_2 = tf.concat([refdecode5_1,refencode1_3], 4)
            refdecode5_4 = relu(
                tf.layers.batch_normalization(conv3d(refdecode5_2, 64, 32, 3, 1, 'decode5_4'), training=True, name='bnde5_4'))
            refdecode5_5 = relu(
                tf.layers.batch_normalization(conv3d(refdecode5_4, 32, 32, 3, 1, 'decode5_5'), training=True, name='bnde5_5'))
            refdecode5_6 = relu(
                tf.layers.batch_normalization(conv3d(refdecode5_5, 32, 32, 3, 1, 'decode5_6') + refdecode5_4, training=True,
                                              name='bnde5_6'))
            refcosout = conv3d_b(refdecode5_6, 32, 1, 3, 1, 'out') + cosout
        return cosout,refcosout
def synthesis_nogp_ref(maginput,fieldinput,tkdinput):
    with tf.variable_scope('syn'):
        with tf.variable_scope('rec'):
            # input2=tf.concat([maginput,fieldinput,tkdinput],4)
            # input2_112= tf.reshape(input2,[32,32,32])
            # input2_56 = tf.image.resize_images(input2, [16, 16,16])
            # input2_28 = tf.image.resize_images(input2, [8, 8,8])
            magconv1 = relu(
                tf.layers.batch_normalization(conv3d(maginput, 1, 48, 3, 1, 'magconv1'), training=True, name='bnmag1'))
            magconv2_33 = relu(
                tf.layers.batch_normalization(conv3d(magconv1, 48, 16, 3, 1, 'magconv2_33'), training=True,
                                              name='bnmag2_33'))
            magconv2_11 = relu(
                tf.layers.batch_normalization(conv3d(magconv1, 48, 16, 1, 1, 'magconv2_11'), training=True,
                                              name='bnmag2_11'))
            magconv2_55 = relu(
                tf.layers.batch_normalization(conv3d(magconv1, 48, 16, 5, 1, 'magconv2_55'), training=True,
                                              name='bnmag2_55'))
            magcat1 = tf.concat([magconv2_33, magconv2_11, magconv2_55], 4)

            magconv3_33 = relu(
                tf.layers.batch_normalization(conv3d(magcat1, 48, 16, 3, 1, 'magconv3_33'), training=True,
                                              name='bnmag3_33'))
            magconv3_11 = relu(
                tf.layers.batch_normalization(conv3d(magcat1, 48, 16, 1, 1, 'magconv3_11'), training=True,
                                              name='bnmag3_11'))
            magconv3_55 = relu(
                tf.layers.batch_normalization(conv3d(magcat1, 48, 16, 5, 1, 'magconv3_55'), training=True,
                                              name='bnmag3_55'))
            magcat2 = tf.concat([magconv3_33, magconv3_11, magconv3_55], 4)

            magconv4_33 = relu(
                tf.layers.batch_normalization(conv3d(magcat2, 48, 16, 3, 1, 'magconv4_33'), training=True,
                                              name='bnmag4_33'))
            magconv4_11 = relu(
                tf.layers.batch_normalization(conv3d(magcat2, 48, 16, 1, 1, 'magconv4_11'), training=True,
                                              name='bnmag4_11'))
            magconv4_55 = relu(
                tf.layers.batch_normalization(conv3d(magcat2, 48, 16, 5, 1, 'magconv4_55'), training=True,
                                              name='bnmag4_55'))
            magcat3 = tf.concat([magconv4_33, magconv4_11, magconv4_55], 4)

            # magconv5_33 = relu(tf.layers.batch_normalization(conv3d(magcat3, 48, 16, 3, 1, 'magconv5_33'), training=True,name='bnmag5_33'))
            # magconv5_11 = relu(tf.layers.batch_normalization(conv3d(magcat3, 48, 16, 1, 1, 'magconv5_11'), training=True,name='bnmag5_11'))
            # magconv5_55 = relu(tf.layers.batch_normalization(conv3d(magcat3, 48, 16, 5, 1, 'magconv5_55'), training=True,name='bnmag5_55'))
            # magcat4 = tf.concat([magconv5_33, magconv5_11, magconv5_55], 4)


            magconcat = tf.concat([magcat1, magcat2, magcat3], 4)
            magweight = tf.get_variable('magw', shape=[144], dtype=tf.float32,
                                        initializer=tf.ones_initializer(), regularizer=None,
                                        trainable=True, collections=None)
            magzong = magconcat * magweight
            magin = relu(
                tf.layers.batch_normalization(conv3d(magzong, 144, 32, 3, 1, 'magin'), training=True, name='bnmagin'))

            tkdconv1 = relu(
                tf.layers.batch_normalization(conv3d(tkdinput, 1, 48, 3, 1, 'tkdconv1'), training=True, name='bntkd1'))
            tkdconv2_33 = relu(
                tf.layers.batch_normalization(conv3d(tkdconv1, 48, 16, 3, 1, 'tkdconv2_33'), training=True,
                                              name='bntkd2_33'))
            tkdconv2_11 = relu(
                tf.layers.batch_normalization(conv3d(tkdconv1, 48, 16, 1, 1, 'tkdconv2_11'), training=True,
                                              name='bntkd2_11'))
            tkdconv2_55 = relu(
                tf.layers.batch_normalization(conv3d(tkdconv1, 48, 16, 5, 1, 'tkdconv2_55'), training=True,
                                              name='bntkd2_55'))
            tkdcat1 = tf.concat([tkdconv2_33, tkdconv2_11, tkdconv2_55], 4)

            tkdconv3_33 = relu(
                tf.layers.batch_normalization(conv3d(tkdcat1, 48, 16, 3, 1, 'tkdconv3_33'), training=True,
                                              name='bntkd3_33'))
            tkdconv3_11 = relu(
                tf.layers.batch_normalization(conv3d(tkdcat1, 48, 16, 1, 1, 'tkdconv3_11'), training=True,
                                              name='bntkd3_11'))
            tkdconv3_55 = relu(
                tf.layers.batch_normalization(conv3d(tkdcat1, 48, 16, 5, 1, 'tkdconv3_55'), training=True,
                                              name='bntkd3_55'))
            tkdcat2 = tf.concat([tkdconv3_33, tkdconv3_11, tkdconv3_55], 4)

            tkdconv4_33 = relu(
                tf.layers.batch_normalization(conv3d(tkdcat2, 48, 16, 3, 1, 'tkdconv4_33'), training=True,
                                              name='bntkd4_33'))
            tkdconv4_11 = relu(
                tf.layers.batch_normalization(conv3d(tkdcat2, 48, 16, 1, 1, 'tkdconv4_11'), training=True,
                                              name='bntkd4_11'))
            tkdconv4_55 = relu(
                tf.layers.batch_normalization(conv3d(tkdcat2, 48, 16, 5, 1, 'tkdconv4_55'), training=True,
                                              name='bntkd4_55'))
            tkdcat3 = tf.concat([tkdconv4_33, tkdconv4_11, tkdconv4_55], 4)

            # tkdconv5_33 = relu(tf.layers.batch_normalization(conv3d(tkdcat3, 48, 16, 3, 1, 'tkdconv5_33'), training=True,name='bntkd5_33'))
            # tkdconv5_11 = relu(tf.layers.batch_normalization(conv3d(tkdcat3, 48, 16, 1, 1, 'tkdconv5_11'), training=True,name='bntkd5_11'))
            # tkdconv5_55 = relu(tf.layers.batch_normalization(conv3d(tkdcat3, 48, 16, 5, 1, 'tkdconv5_55'), training=True,name='bntkd5_55'))
            # tkdcat4 = tf.concat([tkdconv5_33, tkdconv5_11, tkdconv5_55], 4)

            tkdconcat = tf.concat([tkdcat1, tkdcat2, tkdcat3], 4)
            tkdweight = tf.get_variable('tkdw', shape=[144], dtype=tf.float32,
                                        initializer=tf.ones_initializer(), regularizer=None,
                                        trainable=True, collections=None)
            tkdzong = tkdconcat * tkdweight
            tkdin = relu(
                tf.layers.batch_normalization(conv3d(tkdzong, 144, 32, 3, 1, 'tkdin'), training=True, name='bntkdin'))

            fieldconv1 = relu(
                tf.layers.batch_normalization(conv3d(fieldinput, 1, 48, 3, 1, 'fieldconv1'), training=True,
                                              name='bnfield1'))
            fieldconv2_33 = relu(
                tf.layers.batch_normalization(conv3d(fieldconv1, 48, 16, 3, 1, 'fieldconv2_33'), training=True,
                                              name='bnfield2_33'))
            fieldconv2_11 = relu(
                tf.layers.batch_normalization(conv3d(fieldconv1, 48, 16, 1, 1, 'fieldconv2_11'), training=True,
                                              name='bnfield2_11'))
            fieldconv2_55 = relu(
                tf.layers.batch_normalization(conv3d(fieldconv1, 48, 16, 5, 1, 'fieldconv2_55'), training=True,
                                              name='bnfield2_55'))
            fieldcat1 = tf.concat([fieldconv2_33, fieldconv2_11, fieldconv2_55], 4)

            fieldconv3_33 = relu(
                tf.layers.batch_normalization(conv3d(fieldcat1, 48, 16, 3, 1, 'fieldconv3_33'), training=True,
                                              name='bnfield3_33'))
            fieldconv3_11 = relu(
                tf.layers.batch_normalization(conv3d(fieldcat1, 48, 16, 1, 1, 'fieldconv3_11'), training=True,
                                              name='bnfield3_11'))
            fieldconv3_55 = relu(
                tf.layers.batch_normalization(conv3d(fieldcat1, 48, 16, 5, 1, 'fieldconv3_55'), training=True,
                                              name='bnfield3_55'))
            fieldcat2 = tf.concat([fieldconv3_33, fieldconv3_11, fieldconv3_55], 4)

            fieldconv4_33 = relu(
                tf.layers.batch_normalization(conv3d(fieldcat2, 48, 16, 3, 1, 'fieldconv4_33'), training=True,
                                              name='bnfield4_33'))
            fieldconv4_11 = relu(
                tf.layers.batch_normalization(conv3d(fieldcat2, 48, 16, 1, 1, 'fieldconv4_11'), training=True,
                                              name='bnfield4_11'))
            fieldconv4_55 = relu(
                tf.layers.batch_normalization(conv3d(fieldcat2, 48, 16, 5, 1, 'fieldconv4_55'), training=True,
                                              name='bnfield4_55'))
            fieldcat3 = tf.concat([fieldconv4_33, fieldconv4_11, fieldconv4_55], 4)

            # fieldconv5_33 = relu(tf.layers.batch_normalization(conv3d(fieldcat3, 48, 16, 3, 1, 'fieldconv5_33'), training=True,name='bnfield5_33'))
            # fieldconv5_11 = relu(tf.layers.batch_normalization(conv3d(fieldcat3, 48, 16, 1, 1, 'fieldconv5_11'), training=True,name='bnfield5_11'))
            # fieldconv5_55 = relu(tf.layers.batch_normalization(conv3d(fieldcat3, 48, 16, 5, 1, 'fieldconv5_55'), training=True,name='bnfield5_55'))
            # fieldcat4 = tf.concat([fieldconv5_33, fieldconv5_11, fieldconv5_55], 4)

            fieldconcat = tf.concat([fieldcat1, fieldcat2, fieldcat3], 4)
            fieldweight = tf.get_variable('fieldw', shape=[144], dtype=tf.float32,
                                          initializer=tf.ones_initializer(), regularizer=None,
                                          trainable=True, collections=None)
            fieldzong = fieldconcat * fieldweight
            fieldin = relu(tf.layers.batch_normalization(conv3d(fieldzong, 144, 32, 3, 1, 'fieldin'), training=True,
                                                         name='bnfieldin'))

            encodeinput = tf.concat([magin, fieldin, tkdin], 4)
            encode1_1 = relu(
                tf.layers.batch_normalization(conv3d(encodeinput, 96, 32, 3, 1, 'encode1_1'), training=True,
                                              name='bnen1'))
            encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=True,
                                                           name='bnen1_2'))
            encode1_3 = relu(
                tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 1, 'encode1_3') + encode1_1, training=True,
                                              name='bnen1_3'))

            encode2_1 = max_pool3d(encode1_3)
            encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=True,
                                                           name='bnen2_2'))
            encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=True,
                                                           name='bnen2_3'))
            encode2_4 = relu(
                tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 1, 'encode2_4') + encode2_2, training=True,
                                              name='bnen2_4'))

            encode3_1 = max_pool3d(encode2_4)
            encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=True,
                                                           name='bnen3_2'))
            encode3_3 = relu(
                tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=True,
                                              name='bnen3_3'))
            encode3_4 = relu(
                tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 1, 'encode3_4') + encode3_2, training=True,
                                              name='bnen3_4'))

            encode4_1 = max_pool3d(encode3_4)
            encode4_3 = relu(
                tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_3'), training=True,
                                              name='bnen4_3'))
            encode4_4 = relu(
                tf.layers.batch_normalization(conv3d(encode4_3, 256, 256, 3, 1, 'encode4_4'), training=True,
                                              name='bnen4_4'))
            encode4_5 = relu(
                tf.layers.batch_normalization(conv3d(encode4_4, 256, 256, 3, 1, 'encode4_5') + encode4_3, training=True,
                                              name='bnen4_5'))

            decode3_1 = relu(
                tf.layers.batch_normalization(deconv3d(encode4_5, 256, 128, 3, 2, 'decode3_1'), training=True,
                                              name='bnde3_1'))
            decode3_2 = tf.concat([decode3_1, encode3_4], 4)
            decode3_4 = relu(
                tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=True,
                                              name='bnde3_4'))
            decode3_5 = relu(
                tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_5'), training=True,
                                              name='bnde3_5'))
            decode3_6 = relu(
                tf.layers.batch_normalization(conv3d(decode3_5, 128, 128, 3, 1, 'decode3_6') + decode3_4, training=True,
                                              name='bnde3_6'))

            decode4_1 = relu(
                tf.layers.batch_normalization(deconv3d(decode3_6, 128, 64, 3, 2, 'decode2_1'), training=True,
                                              name='bnde2_1'))
            decode4_2 = tf.concat([decode4_1, encode2_4], 4)
            decode4_4 = relu(tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=True,
                                                           name='bnde4_4'))
            decode4_5 = relu(tf.layers.batch_normalization(conv3d(decode4_4, 64, 64, 3, 1, 'decode4_5'), training=True,
                                                           name='bnde4_5'))
            decode4_6 = relu(
                tf.layers.batch_normalization(conv3d(decode4_5, 64, 64, 3, 1, 'decode4_6') + decode4_4, training=True,
                                              name='bnde4_6'))

            decode5_1 = relu(
                tf.layers.batch_normalization(deconv3d(decode4_6, 64, 32, 3, 2, 'decode1_1'), training=True,
                                              name='bnde1_1'))
            decode5_2 = tf.concat([decode5_1, encode1_3], 4)
            decode5_4 = relu(tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=True,
                                                           name='bnde5_4'))
            decode5_5 = relu(tf.layers.batch_normalization(conv3d(decode5_4, 32, 32, 3, 1, 'decode5_5'), training=True,
                                                           name='bnde5_5'))
            decode5_6 = relu(
                tf.layers.batch_normalization(conv3d(decode5_5, 32, 32, 3, 1, 'decode5_6') + decode5_4, training=True,
                                              name='bnde5_6'))
            cosout = conv3d_b(decode5_6, 32, 1, 3, 1, 'out') + tkdinput



        with tf.variable_scope('ref'):
            fea=Feature_Extraction(cosout)
            refencodeinput = tf.concat([cosout, fea], 4)
            refencode1_1 = relu(
                tf.layers.batch_normalization(conv3d(refencodeinput, 10, 32, 3, 1, 'encode1_1'), training=True, name='bnen1'))
            refencode1_2 = relu(
                tf.layers.batch_normalization(conv3d(refencode1_1, 32, 32, 3, 1, 'encode1_2'), training=True, name='bnen1_2'))
            refencode1_3 = relu(
                tf.layers.batch_normalization(conv3d(refencode1_2, 32, 32, 3, 1, 'encode1_3') + refencode1_1, training=True,
                                              name='bnen1_3'))

            refencode2_1 = max_pool3d(refencode1_3)
            refencode2_2 = relu(
                tf.layers.batch_normalization(conv3d(refencode2_1, 32, 64, 3, 1, 'encode2_2'), training=True, name='bnen2_2'))
            refencode2_3 = relu(
                tf.layers.batch_normalization(conv3d(refencode2_2, 64, 64, 3, 1, 'encode2_3'), training=True, name='bnen2_3'))
            refencode2_4 = relu(
                tf.layers.batch_normalization(conv3d(refencode2_3, 64, 64, 3, 1, 'encode2_4') + refencode2_2, training=True,
                                              name='bnen2_4'))

            refencode3_1 = max_pool3d(refencode2_4)
            refencode3_2 = relu(
                tf.layers.batch_normalization(conv3d(refencode3_1, 64, 128, 3, 1, 'encode3_2'), training=True, name='bnen3_2'))
            refencode3_3 = relu(tf.layers.batch_normalization(conv3d(refencode3_2, 128, 128, 3, 1, 'encode3_3'), training=True,
                                                           name='bnen3_3'))
            refencode3_4 = relu(
                tf.layers.batch_normalization(conv3d(refencode3_3, 128, 128, 3, 1, 'encode3_4') + refencode3_2, training=True,
                                              name='bnen3_4'))

            refencode4_1 = max_pool3d(refencode3_4)
            refencode4_3 = relu(tf.layers.batch_normalization(conv3d(refencode4_1, 128, 256, 3, 1, 'encode4_3'), training=True,
                                                           name='bnen4_3'))
            refencode4_4 = relu(tf.layers.batch_normalization(conv3d(refencode4_3, 256, 256, 3, 1, 'encode4_4'), training=True,
                                                           name='bnen4_4'))
            refencode4_5 = relu(
                tf.layers.batch_normalization(conv3d(refencode4_4, 256, 256, 3, 1, 'encode4_5') + refencode4_3, training=True,
                                              name='bnen4_5'))

            refdecode3_1 = relu(tf.layers.batch_normalization(deconv3d(refencode4_5, 256, 128, 3, 2, 'decode3_1'), training=True,
                                                           name='bnde3_1'))
            refdecode3_2 = tf.concat([refdecode3_1,refencode3_4], 4)
            refdecode3_4 = relu(tf.layers.batch_normalization(conv3d(refdecode3_2, 256, 128, 3, 1, 'decode3_4'), training=True,
                                                           name='bnde3_4'))
            refdecode3_5 = relu(tf.layers.batch_normalization(conv3d(refdecode3_4, 128, 128, 3, 1, 'decode3_5'), training=True,
                                                           name='bnde3_5'))
            refdecode3_6 = relu(
                tf.layers.batch_normalization(conv3d(refdecode3_5, 128, 128, 3, 1, 'decode3_6') + refdecode3_4, training=True,
                                              name='bnde3_6'))

            refdecode4_1 = relu(tf.layers.batch_normalization(deconv3d(refdecode3_6, 128, 64, 3, 2, 'decode2_1'), training=True,
                                                           name='bnde2_1'))
            refdecode4_2 = tf.concat([refdecode4_1,refencode2_4], 4)
            refdecode4_4 = relu(
                tf.layers.batch_normalization(conv3d(refdecode4_2, 128, 64, 3, 1, 'decode4_4'), training=True, name='bnde4_4'))
            refdecode4_5 = relu(
                tf.layers.batch_normalization(conv3d(refdecode4_4, 64, 64, 3, 1, 'decode4_5'), training=True, name='bnde4_5'))
            refdecode4_6 = relu(
                tf.layers.batch_normalization(conv3d(refdecode4_5, 64, 64, 3, 1, 'decode4_6') + refdecode4_4, training=True,
                                              name='bnde4_6'))

            refdecode5_1 = relu(tf.layers.batch_normalization(deconv3d(refdecode4_6, 64, 32, 3, 2, 'decode1_1'), training=True,
                                                           name='bnde1_1'))
            refdecode5_2 = tf.concat([refdecode5_1,refencode1_3], 4)
            refdecode5_4 = relu(
                tf.layers.batch_normalization(conv3d(refdecode5_2, 64, 32, 3, 1, 'decode5_4'), training=True, name='bnde5_4'))
            refdecode5_5 = relu(
                tf.layers.batch_normalization(conv3d(refdecode5_4, 32, 32, 3, 1, 'decode5_5'), training=True, name='bnde5_5'))
            refdecode5_6 = relu(
                tf.layers.batch_normalization(conv3d(refdecode5_5, 32, 32, 3, 1, 'decode5_6') + refdecode5_4, training=True,
                                              name='bnde5_6'))
            refcosout = conv3d_b(refdecode5_6, 32, 1, 3, 1, 'out') + cosout
    return cosout,refcosout
def t_synthesis_newarc(maginput,fieldinput,tkdinput):
    with tf.variable_scope('syn'):
        #input2=tf.concat([maginput,fieldinput,tkdinput],4)
        # input2_112= tf.reshape(input2,[32,32,32])
        # input2_56 = tf.image.resize_images(input2, [16, 16,16])
        # input2_28 = tf.image.resize_images(input2, [8, 8,8])
        magconv1=relu(tf.layers.batch_normalization(conv3d(maginput,1,17,5,1,'magconv1'), training=True, name='bnmag1'))
        weight = tf.get_variable('w', shape=[17], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),
                                 regularizer=None, trainable=True, collections=None)
        a=magconv1*weight
        magconv2_1 = relu(tf.layers.batch_normalization(conv3d(magconv1, 16, 16, 3, 1, 'magconv2_1'), training=True, name='bnmag2_1'))
        magconv2_2 = relu(tf.layers.batch_normalization(conv3d(magconv2_1, 16, 16, 3, 1, 'magconv2_2')+magconv1, training=True, name='bnmag2_2'))
        magconv3_1 = relu(tf.layers.batch_normalization(conv3d(magconv2_2, 16, 16, 3, 1, 'magconv3_1'), training=True, name='bnmag3_1'))
        magconv3_2 = relu(tf.layers.batch_normalization(conv3d(magconv3_1, 16, 16, 3, 1, 'magconv3_2') + magconv2_2, training=True, name='bnmag3_2'))
        magconv4_1 = relu(tf.layers.batch_normalization(conv3d(magconv3_2, 16, 16, 3, 1, 'magconv4_1'), training=True, name='bnmag4_1'))
        magconv4_2 = relu(tf.layers.batch_normalization(conv3d(magconv4_1, 16, 16, 3, 1, 'magconv4_2') + magconv3_2, training=True, name='bnmag4_2'))
        magconv5_1 = relu(tf.layers.batch_normalization(conv3d(magconv4_2, 16, 16, 3, 1, 'magconv5_1'), training=True, name='bnmag5_1'))
        magconv5_2 = relu(tf.layers.batch_normalization(conv3d(magconv5_1, 16, 16, 3, 1, 'magconv5_2') + magconv4_2, training=True, name='bnmag5_2'))
        magconv6_1 = relu(tf.layers.batch_normalization(conv3d(magconv5_2, 16, 16, 3, 2, 'magconv6_1'), training=True,name='bnmag6_1'))
        magconv6_2 = relu(tf.layers.batch_normalization(conv3d(magconv6_1, 16, 16, 3, 1, 'magconv6_2'), training=True,name='bnmag6_2'))
        magconv7_1 = relu(tf.layers.batch_normalization(conv3d(magconv6_2, 16, 16, 3, 2, 'magconv7_1'), training=True,name='bnmag7_1'))



        tkdconv1 = relu(
            tf.layers.batch_normalization(conv3d(tkdinput, 1, 16, 5, 1, 'tkdconv1'), training=True, name='bntkd1'))
        tkdconv2_1 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv1, 16, 16, 3, 1, 'tkdconv2_1'), training=True, name='bntkd2_1'))
        tkdconv2_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv2_1, 16, 16, 3, 1, 'tkdconv2_2') + tkdconv1, training=True,
                                          name='bntkd2_2'))
        tkdconv3_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv2_2, 16, 16, 3, 1, 'tkdconv3_1'), training=True,
                                                        name='bntkd3_1'))
        tkdconv3_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv3_1, 16, 16, 3, 1, 'tkdconv3_2') + tkdconv2_2, training=True,
                                          name='bntkd3_2'))
        tkdconv4_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv3_2, 16, 16, 3, 1, 'tkdconv4_1'), training=True,
                                                        name='bntkd4_1'))
        tkdconv4_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv4_1, 16, 16, 3, 1, 'tkdconv4_2') + tkdconv3_2, training=True,
                                          name='bntkd4_2'))
        tkdconv5_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv4_2, 16, 16, 3, 1, 'tkdonv5_1'), training=True,
                                                        name='bntkd5_1'))
        tkdconv5_2 = relu(
            tf.layers.batch_normalization(conv3d(tkdconv5_1, 16, 16, 3, 1, 'tkdconv5_2') + tkdconv4_2, training=True,
                                          name='bntkd5_2'))
        tkdconv6_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv5_2, 16, 16, 3, 2, 'tkdconv6_1'), training=True,name='bntkd6_1'))
        tkdconv6_2 = relu(tf.layers.batch_normalization(conv3d(tkdconv6_1, 16, 16, 3, 1, 'tkdconv6_2'), training=True,name='bntkd6_2'))
        tkdconv7_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv6_2, 16, 16, 3, 2, 'tkdconv7_1'), training=True,name='bntkd7_1'))



        fieldconv1=relu(tf.layers.batch_normalization(conv3d(fieldinput,1,16,5,1,'fieldconv1'), training=True, name='bnfield1'))
        fieldconv2_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv1, 16, 16, 3, 1, 'fieldconv2_1'), training=True, name='bndield2_1'))
        fieldconv2_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_1, 16, 16, 3, 1, 'fieldconv2_2')+fieldconv1, training=True, name='bnfield2_2'))
        fieldconv3_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_2, 16, 16, 3, 1, 'fieldconv3_1'), training=True, name='bnfield3_1'))
        fieldconv3_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_1, 16, 16, 3, 1, 'fieldconv3_2') + fieldconv2_2, training=True, name='bnfield3_2'))
        fieldconv4_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_2, 16, 16, 3, 1, 'fieldconv4_1'), training=True, name='bnfield4_1'))
        fieldconv4_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_1, 16, 16, 3, 1, 'fieldconv4_2') + fieldconv3_2, training=True, name='bnfield4_2'))
        fieldconv5_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_2, 16, 16, 3, 1, 'fieldconv5_1'), training=True, name='bnfield5_1'))
        fieldconv5_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv5_1, 16, 16, 3, 1, 'fieldconv5_2') + fieldconv4_2, training=True, name='bnfield5_2'))
        fieldconv6_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv5_2, 16, 16, 3, 2, 'fieldconv6_1'), training=True,name='bnfield6_1'))
        fieldconv6_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv6_1, 16, 16, 3, 1, 'fieldconv6_2'), training=True,name='bnfield6_2'))
        fieldconv7_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv6_2, 16, 16, 3, 2, 'fieldconv7_1'), training=True,name='bnfield7_1'))



        concat7_1=tf.concat([magconv7_1,tkdconv7_1,fieldconv7_1],4)
        globalpool=tf.reduce_mean(concat7_1,[2,3],keep_dims=True)
        fusionflatten = tf.reshape(globalpool, (1, 192))
        magdense = tf.layers.dense(fusionflatten, 256)
        tkddense = tf.layers.dense(fusionflatten, 256)
        fielddense = tf.layers.dense(fusionflatten, 256)

        magweight = (tf.reshape(magdense, (1, 16, 1, 1, 16)))
        magin = magconv5_2 * magweight
        tkdweight = (tf.reshape(tkddense, (1, 16, 1, 1, 16)))
        tkdin = tkdconv5_2 * tkdweight
        fieldweight = (tf.reshape(fielddense, (1, 16, 1, 1, 16)))
        fieldin = fieldconv5_2 * fieldweight

        encodeinput=tf.concat([magin,fieldin,tkdin],4)
        encode1_1=relu(tf.layers.batch_normalization(conv3d(encodeinput,48,32,3,1,'encode1_1'), training=True, name='bnen1'))
        encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=True, name='bnen1_2'))
        encode1_3 = relu(tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 1, 'encode1_3')+encode1_1, training=True, name='bnen1_3'))

        encode2_1=max_pool3d(encode1_3)
        encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=True, name='bnen2_2'))
        encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=True, name='bnen2_3'))
        encode2_4 = relu(tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 1, 'encode2_4')+encode2_2, training=True, name='bnen2_4'))

        encode3_1=max_pool3d(encode2_4)
        encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=True, name='bnen3_2'))
        encode3_3 = relu(tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=True, name='bnen3_3'))
        encode3_4 = relu(tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 1, 'encode3_4')+encode3_2, training=True, name='bnen3_4'))

        encode4_1 = max_pool3d(encode3_4)
        encode4_3 = relu(tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_3'), training=True, name='bnen4_3'))
        encode4_4 = relu(tf.layers.batch_normalization(conv3d(encode4_3, 256, 256, 3, 1, 'encode4_4'), training=True, name='bnen4_4'))
        encode4_5 = relu(tf.layers.batch_normalization(conv3d(encode4_4, 256, 256, 3, 1, 'encode4_5')+encode4_3, training=True, name='bnen4_5'))

        decode3_1 = relu(tf.layers.batch_normalization(deconv3d(encode4_5, 256, 128, 3, 2, 'decode3_1'), training=True,
                                                       name='bnde3_1'))
        decode3_2=tf.concat([decode3_1,encode3_4],4)
        decode3_4 = relu(tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=True, name='bnde3_4'))
        decode3_5 = relu(tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_5'), training=True, name='bnde3_5'))
        decode3_6 = relu(tf.layers.batch_normalization(conv3d(decode3_5, 128, 128, 3, 1, 'decode3_6')+decode3_4, training=True, name='bnde3_6'))

        decode4_1 = relu(tf.layers.batch_normalization(deconv3d(decode3_6, 128, 64, 3, 2, 'decode2_1'), training=True,
                                          name='bnde2_1'))
        decode4_2 = tf.concat([decode4_1,encode2_4],4)
        decode4_4 = relu(tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=True, name='bnde4_4'))
        decode4_5 = relu(tf.layers.batch_normalization(conv3d(decode4_4, 64, 64, 3, 1, 'decode4_5'), training=True, name='bnde4_5'))
        decode4_6 = relu(tf.layers.batch_normalization(conv3d(decode4_5, 64, 64, 3, 1, 'decode4_6')+decode4_4, training=True, name='bnde4_6'))

        decode5_1 = relu(tf.layers.batch_normalization(deconv3d(decode4_6, 64, 32, 3, 2, 'decode1_1'), training=True,
                                                       name='bnde1_1'))
        decode5_2 = tf.concat([decode5_1,encode1_3],4)
        decode5_4 = relu(tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=True, name='bnde5_4'))
        decode5_5 = relu(tf.layers.batch_normalization(conv3d(decode5_4, 32, 32, 3, 1, 'decode5_5'), training=True, name='bnde5_5'))
        decode5_6 = relu(tf.layers.batch_normalization(conv3d(decode5_5, 32, 32, 3, 1, 'decode5_6')+decode5_4, training=True, name='bnde5_6'))
        cosout=conv3d_b(decode5_6, 32, 1, 3, 1, 'out')+tkdinput


        return cosout
def t_synthesis_newarc_ref(maginput,fieldinput,tkdinput):
    with tf.variable_scope('syn'):
        with tf.variable_scope('rec'):
            #input2=tf.concat([maginput,fieldinput,tkdinput],4)
            # input2_112= tf.reshape(input2,[32,32,32])
            # input2_56 = tf.image.resize_images(input2, [16, 16,16])
            # input2_28 = tf.image.resize_images(input2, [8, 8,8])
            magconv1=relu(tf.layers.batch_normalization(conv3d(maginput,1,16,5,1,'magconv1'), training=True, name='bnmag1'))
            magconv2_1 = relu(tf.layers.batch_normalization(conv3d(magconv1, 16, 16, 3, 1, 'magconv2_1'), training=True, name='bnmag2_1'))
            magconv2_2 = relu(tf.layers.batch_normalization(conv3d(magconv2_1, 16, 16, 3, 1, 'magconv2_2')+magconv1, training=True, name='bnmag2_2'))
            magconv3_1 = relu(tf.layers.batch_normalization(conv3d(magconv2_2, 16, 16, 3, 1, 'magconv3_1'), training=True, name='bnmag3_1'))
            magconv3_2 = relu(tf.layers.batch_normalization(conv3d(magconv3_1, 16, 16, 3, 1, 'magconv3_2') + magconv2_2, training=True, name='bnmag3_2'))
            magconv4_1 = relu(tf.layers.batch_normalization(conv3d(magconv3_2, 16, 16, 3, 1, 'magconv4_1'), training=True, name='bnmag4_1'))
            magconv4_2 = relu(tf.layers.batch_normalization(conv3d(magconv4_1, 16, 16, 3, 1, 'magconv4_2') + magconv3_2, training=True, name='bnmag4_2'))
            magconv5_1 = relu(tf.layers.batch_normalization(conv3d(magconv4_2, 16, 16, 3, 1, 'magconv5_1'), training=True, name='bnmag5_1'))
            magconv5_2 = relu(tf.layers.batch_normalization(conv3d(magconv5_1, 16, 16, 3, 1, 'magconv5_2') + magconv4_2, training=True, name='bnmag5_2'))
            magconv6_1 = relu(tf.layers.batch_normalization(conv3d(magconv5_2, 16, 16, 3, 2, 'magconv6_1'), training=True,name='bnmag6_1'))
            magconv6_2 = relu(tf.layers.batch_normalization(conv3d(magconv6_1, 16, 16, 3, 1, 'magconv6_2'), training=True,name='bnmag6_2'))
            magconv7_1 = relu(tf.layers.batch_normalization(conv3d(magconv6_2, 16, 16, 3, 2, 'magconv7_1'), training=True,name='bnmag7_1'))



            tkdconv1 = relu(
                tf.layers.batch_normalization(conv3d(tkdinput, 1, 16, 5, 1, 'tkdconv1'), training=True, name='bntkd1'))
            tkdconv2_1 = relu(
                tf.layers.batch_normalization(conv3d(tkdconv1, 16, 16, 3, 1, 'tkdconv2_1'), training=True, name='bntkd2_1'))
            tkdconv2_2 = relu(
                tf.layers.batch_normalization(conv3d(tkdconv2_1, 16, 16, 3, 1, 'tkdconv2_2') + tkdconv1, training=True,
                                              name='bntkd2_2'))
            tkdconv3_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv2_2, 16, 16, 3, 1, 'tkdconv3_1'), training=True,
                                                            name='bntkd3_1'))
            tkdconv3_2 = relu(
                tf.layers.batch_normalization(conv3d(tkdconv3_1, 16, 16, 3, 1, 'tkdconv3_2') + tkdconv2_2, training=True,
                                              name='bntkd3_2'))
            tkdconv4_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv3_2, 16, 16, 3, 1, 'tkdconv4_1'), training=True,
                                                            name='bntkd4_1'))
            tkdconv4_2 = relu(
                tf.layers.batch_normalization(conv3d(tkdconv4_1, 16, 16, 3, 1, 'tkdconv4_2') + tkdconv3_2, training=True,
                                              name='bntkd4_2'))
            tkdconv5_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv4_2, 16, 16, 3, 1, 'tkdonv5_1'), training=True,
                                                            name='bntkd5_1'))
            tkdconv5_2 = relu(
                tf.layers.batch_normalization(conv3d(tkdconv5_1, 16, 16, 3, 1, 'tkdconv5_2') + tkdconv4_2, training=True,
                                              name='bntkd5_2'))
            tkdconv6_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv5_2, 16, 16, 3, 2, 'tkdconv6_1'), training=True,name='bntkd6_1'))
            tkdconv6_2 = relu(tf.layers.batch_normalization(conv3d(tkdconv6_1, 16, 16, 3, 1, 'tkdconv6_2'), training=True,name='bntkd6_2'))
            tkdconv7_1 = relu(tf.layers.batch_normalization(conv3d(tkdconv6_2, 16, 16, 3, 2, 'tkdconv7_1'), training=True,name='bntkd7_1'))



            fieldconv1=relu(tf.layers.batch_normalization(conv3d(fieldinput,1,16,5,1,'fieldconv1'), training=True, name='bnfield1'))
            fieldconv2_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv1, 16, 16, 3, 1, 'fieldconv2_1'), training=True, name='bndield2_1'))
            fieldconv2_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_1, 16, 16, 3, 1, 'fieldconv2_2')+fieldconv1, training=True, name='bnfield2_2'))
            fieldconv3_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv2_2, 16, 16, 3, 1, 'fieldconv3_1'), training=True, name='bnfield3_1'))
            fieldconv3_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_1, 16, 16, 3, 1, 'fieldconv3_2') + fieldconv2_2, training=True, name='bnfield3_2'))
            fieldconv4_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv3_2, 16, 16, 3, 1, 'fieldconv4_1'), training=True, name='bnfield4_1'))
            fieldconv4_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_1, 16, 16, 3, 1, 'fieldconv4_2') + fieldconv3_2, training=True, name='bnfield4_2'))
            fieldconv5_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv4_2, 16, 16, 3, 1, 'fieldconv5_1'), training=True, name='bnfield5_1'))
            fieldconv5_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv5_1, 16, 16, 3, 1, 'fieldconv5_2') + fieldconv4_2, training=True, name='bnfield5_2'))
            fieldconv6_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv5_2, 16, 16, 3, 2, 'fieldconv6_1'), training=True,name='bnfield6_1'))
            fieldconv6_2 = relu(tf.layers.batch_normalization(conv3d(fieldconv6_1, 16, 16, 3, 1, 'fieldconv6_2'), training=True,name='bnfield6_2'))
            fieldconv7_1 = relu(tf.layers.batch_normalization(conv3d(fieldconv6_2, 16, 16, 3, 2, 'fieldconv7_1'), training=True,name='bnfield7_1'))



            concat7_1=tf.concat([magconv7_1,tkdconv7_1,fieldconv7_1],4)
            globalpool = tf.reduce_mean(concat7_1, [2, 3], keep_dims=True)
            fusionflatten = tf.reshape(globalpool, (1, 192))
            magdense = tf.layers.dense(fusionflatten, 256)
            tkddense = tf.layers.dense(fusionflatten, 256)
            fielddense = tf.layers.dense(fusionflatten, 256)

            magweight = (tf.reshape(magdense, (1, 16, 1, 1, 16)))
            magin = magconv5_2 * magweight
            tkdweight = (tf.reshape(tkddense, (1, 16, 1, 1, 16)))
            tkdin = tkdconv5_2 * tkdweight
            fieldweight = (tf.reshape(fielddense, (1, 16, 1, 1, 16)))
            fieldin = fieldconv5_2 * fieldweight

            encodeinput = tf.concat([magin, fieldin, tkdin], 4)
            encode1_1 = relu(
                tf.layers.batch_normalization(conv3d(encodeinput, 48, 32, 3, 1, 'encode1_1'), training=True,
                                              name='bnen1'))
            encode1_2 = relu(tf.layers.batch_normalization(conv3d(encode1_1, 32, 32, 3, 1, 'encode1_2'), training=True,
                                                           name='bnen1_2'))
            encode1_3 = relu(
                tf.layers.batch_normalization(conv3d(encode1_2, 32, 32, 3, 1, 'encode1_3') + encode1_1, training=True,
                                              name='bnen1_3'))

            encode2_1 = max_pool3d(encode1_3)
            encode2_2 = relu(tf.layers.batch_normalization(conv3d(encode2_1, 32, 64, 3, 1, 'encode2_2'), training=True,
                                                           name='bnen2_2'))
            encode2_3 = relu(tf.layers.batch_normalization(conv3d(encode2_2, 64, 64, 3, 1, 'encode2_3'), training=True,
                                                           name='bnen2_3'))
            encode2_4 = relu(
                tf.layers.batch_normalization(conv3d(encode2_3, 64, 64, 3, 1, 'encode2_4') + encode2_2, training=True,
                                              name='bnen2_4'))

            encode3_1 = max_pool3d(encode2_4)
            encode3_2 = relu(tf.layers.batch_normalization(conv3d(encode3_1, 64, 128, 3, 1, 'encode3_2'), training=True,
                                                           name='bnen3_2'))
            encode3_3 = relu(
                tf.layers.batch_normalization(conv3d(encode3_2, 128, 128, 3, 1, 'encode3_3'), training=True,
                                              name='bnen3_3'))
            encode3_4 = relu(
                tf.layers.batch_normalization(conv3d(encode3_3, 128, 128, 3, 1, 'encode3_4') + encode3_2, training=True,
                                              name='bnen3_4'))

            encode4_1 = max_pool3d(encode3_4)
            encode4_3 = relu(
                tf.layers.batch_normalization(conv3d(encode4_1, 128, 256, 3, 1, 'encode4_3'), training=True,
                                              name='bnen4_3'))
            encode4_4 = relu(
                tf.layers.batch_normalization(conv3d(encode4_3, 256, 256, 3, 1, 'encode4_4'), training=True,
                                              name='bnen4_4'))
            encode4_5 = relu(
                tf.layers.batch_normalization(conv3d(encode4_4, 256, 256, 3, 1, 'encode4_5') + encode4_3, training=True,
                                              name='bnen4_5'))

            decode3_1 = relu(
                tf.layers.batch_normalization(deconv3d(encode4_5, 256, 128, 3, 2, 'decode3_1'), training=True,
                                              name='bnde3_1'))
            decode3_2 = tf.concat([decode3_1, encode3_4], 4)
            decode3_4 = relu(
                tf.layers.batch_normalization(conv3d(decode3_2, 256, 128, 3, 1, 'decode3_4'), training=True,
                                              name='bnde3_4'))
            decode3_5 = relu(
                tf.layers.batch_normalization(conv3d(decode3_4, 128, 128, 3, 1, 'decode3_5'), training=True,
                                              name='bnde3_5'))
            decode3_6 = relu(
                tf.layers.batch_normalization(conv3d(decode3_5, 128, 128, 3, 1, 'decode3_6') + decode3_4, training=True,
                                              name='bnde3_6'))

            decode4_1 = relu(
                tf.layers.batch_normalization(deconv3d(decode3_6, 128, 64, 3, 2, 'decode2_1'), training=True,
                                              name='bnde2_1'))
            decode4_2 = tf.concat([decode4_1, encode2_4], 4)
            decode4_4 = relu(tf.layers.batch_normalization(conv3d(decode4_2, 128, 64, 3, 1, 'decode4_4'), training=True,
                                                           name='bnde4_4'))
            decode4_5 = relu(tf.layers.batch_normalization(conv3d(decode4_4, 64, 64, 3, 1, 'decode4_5'), training=True,
                                                           name='bnde4_5'))
            decode4_6 = relu(
                tf.layers.batch_normalization(conv3d(decode4_5, 64, 64, 3, 1, 'decode4_6') + decode4_4, training=True,
                                              name='bnde4_6'))

            decode5_1 = relu(
                tf.layers.batch_normalization(deconv3d(decode4_6, 64, 32, 3, 2, 'decode1_1'), training=True,
                                              name='bnde1_1'))
            decode5_2 = tf.concat([decode5_1, encode1_3], 4)
            decode5_4 = relu(tf.layers.batch_normalization(conv3d(decode5_2, 64, 32, 3, 1, 'decode5_4'), training=True,
                                                           name='bnde5_4'))
            decode5_5 = relu(tf.layers.batch_normalization(conv3d(decode5_4, 32, 32, 3, 1, 'decode5_5'), training=True,
                                                           name='bnde5_5'))
            decode5_6 = relu(
                tf.layers.batch_normalization(conv3d(decode5_5, 32, 32, 3, 1, 'decode5_6') + decode5_4, training=True,
                                              name='bnde5_6'))
            cosout = conv3d_b(decode5_6, 32, 1, 3, 1, 'out') + tkdinput
        with tf.variable_scope('ref'):
            fea = Feature_Extraction(cosout)
            refencodeinput = tf.concat([cosout, fea], 4)
            refencode1_1 = relu(
                tf.layers.batch_normalization(conv3d(refencodeinput, 10, 32, 3, 1, 'encode1_1'), training=True,
                                              name='bnen1'))
            refencode1_2 = relu(
                tf.layers.batch_normalization(conv3d(refencode1_1, 32, 32, 3, 1, 'encode1_2'), training=True,
                                              name='bnen1_2'))
            refencode1_3 = relu(
                tf.layers.batch_normalization(conv3d(refencode1_2, 32, 32, 3, 1, 'encode1_3') + refencode1_1,
                                              training=True,
                                              name='bnen1_3'))

            refencode2_1 = max_pool3d(refencode1_3)
            refencode2_2 = relu(
                tf.layers.batch_normalization(conv3d(refencode2_1, 32, 64, 3, 1, 'encode2_2'), training=True,
                                              name='bnen2_2'))
            refencode2_3 = relu(
                tf.layers.batch_normalization(conv3d(refencode2_2, 64, 64, 3, 1, 'encode2_3'), training=True,
                                              name='bnen2_3'))
            refencode2_4 = relu(
                tf.layers.batch_normalization(conv3d(refencode2_3, 64, 64, 3, 1, 'encode2_4') + refencode2_2,
                                              training=True,
                                              name='bnen2_4'))

            refencode3_1 = max_pool3d(refencode2_4)
            refencode3_2 = relu(
                tf.layers.batch_normalization(conv3d(refencode3_1, 64, 128, 3, 1, 'encode3_2'), training=True,
                                              name='bnen3_2'))
            refencode3_3 = relu(
                tf.layers.batch_normalization(conv3d(refencode3_2, 128, 128, 3, 1, 'encode3_3'), training=True,
                                              name='bnen3_3'))
            refencode3_4 = relu(
                tf.layers.batch_normalization(conv3d(refencode3_3, 128, 128, 3, 1, 'encode3_4') + refencode3_2,
                                              training=True,
                                              name='bnen3_4'))

            refencode4_1 = max_pool3d(refencode3_4)
            refencode4_3 = relu(
                tf.layers.batch_normalization(conv3d(refencode4_1, 128, 256, 3, 1, 'encode4_3'), training=True,
                                              name='bnen4_3'))
            refencode4_4 = relu(
                tf.layers.batch_normalization(conv3d(refencode4_3, 256, 256, 3, 1, 'encode4_4'), training=True,
                                              name='bnen4_4'))
            refencode4_5 = relu(
                tf.layers.batch_normalization(conv3d(refencode4_4, 256, 256, 3, 1, 'encode4_5') + refencode4_3,
                                              training=True,
                                              name='bnen4_5'))

            refdecode3_1 = relu(
                tf.layers.batch_normalization(deconv3d(refencode4_5, 256, 128, 3, 2, 'decode3_1'), training=True,
                                              name='bnde3_1'))
            refdecode3_2 = tf.concat([refdecode3_1, refencode3_4], 4)
            refdecode3_4 = relu(
                tf.layers.batch_normalization(conv3d(refdecode3_2, 256, 128, 3, 1, 'decode3_4'), training=True,
                                              name='bnde3_4'))
            refdecode3_5 = relu(
                tf.layers.batch_normalization(conv3d(refdecode3_4, 128, 128, 3, 1, 'decode3_5'), training=True,
                                              name='bnde3_5'))
            refdecode3_6 = relu(
                tf.layers.batch_normalization(conv3d(refdecode3_5, 128, 128, 3, 1, 'decode3_6') + refdecode3_4,
                                              training=True,
                                              name='bnde3_6'))

            refdecode4_1 = relu(
                tf.layers.batch_normalization(deconv3d(refdecode3_6, 128, 64, 3, 2, 'decode2_1'), training=True,
                                              name='bnde2_1'))
            refdecode4_2 = tf.concat([refdecode4_1, refencode2_4], 4)
            refdecode4_4 = relu(
                tf.layers.batch_normalization(conv3d(refdecode4_2, 128, 64, 3, 1, 'decode4_4'), training=True,
                                              name='bnde4_4'))
            refdecode4_5 = relu(
                tf.layers.batch_normalization(conv3d(refdecode4_4, 64, 64, 3, 1, 'decode4_5'), training=True,
                                              name='bnde4_5'))
            refdecode4_6 = relu(
                tf.layers.batch_normalization(conv3d(refdecode4_5, 64, 64, 3, 1, 'decode4_6') + refdecode4_4,
                                              training=True,
                                              name='bnde4_6'))

            refdecode5_1 = relu(
                tf.layers.batch_normalization(deconv3d(refdecode4_6, 64, 32, 3, 2, 'decode1_1'), training=True,
                                              name='bnde1_1'))
            refdecode5_2 = tf.concat([refdecode5_1, refencode1_3], 4)
            refdecode5_4 = relu(
                tf.layers.batch_normalization(conv3d(refdecode5_2, 64, 32, 3, 1, 'decode5_4'), training=True,
                                              name='bnde5_4'))
            refdecode5_5 = relu(
                tf.layers.batch_normalization(conv3d(refdecode5_4, 32, 32, 3, 1, 'decode5_5'), training=True,
                                              name='bnde5_5'))
            refdecode5_6 = relu(
                tf.layers.batch_normalization(conv3d(refdecode5_5, 32, 32, 3, 1, 'decode5_6') + refdecode5_4,
                                              training=True,
                                              name='bnde5_6'))
            refcosout = conv3d_b(refdecode5_6, 32, 1, 3, 1, 'out') + cosout
        return cosout, refcosout
def discriminator(input,reuse):
    with tf.variable_scope('dis',reuse=reuse):
        conv1 = relu(tf.layers.batch_normalization(conv2d(input, 1, 64, 3, 1, 'conv1'), training=True, name='bn1'))
        conv2 = relu(tf.layers.batch_normalization(conv2d(conv1, 64, 24, 1, 1, 'conv2'), training=True, name='bn2'))
        conv2_1 = relu(tf.layers.batch_normalization(conv2d(conv1, 64, 96, 1, 1, 'conv2_1'), training=True, name='bn2_1'))
        conv3 = relu(tf.layers.batch_normalization(conv2d(conv2, 24, 24, 3, 1, 'conv3'), training=True, name='bn3'))
        conv4 = relu(
            tf.layers.batch_normalization(conv2d(conv3, 24, 96, 1, 1, 'conv4'), training=True, name='bn4') + conv2_1)
        pool1 = pool(conv4)

        conv5 = relu(tf.layers.batch_normalization(conv2d(pool1, 96, 48, 1, 1, 'conv5'), training=True, name='bn5'))
        conv5_1 = relu(tf.layers.batch_normalization(conv2d(pool1, 96, 192, 1, 1, 'conv5_1'), training=True, name='bn5_1'))
        conv6 = relu(tf.layers.batch_normalization(conv2d(conv5, 48, 48, 3, 1, 'conv6'), training=True, name='bn6'))
        conv7 = relu(
            tf.layers.batch_normalization(conv2d(conv6, 48, 192, 1, 1, 'conv7'), training=True, name='bn7') + conv5_1)
        pool2 = pool(conv7)
        conv8 = relu(tf.layers.batch_normalization(conv2d(pool2, 192, 96, 1, 1, 'conv8'), training=True, name='bn8'))
        conv8_1 = relu(tf.layers.batch_normalization(conv2d(pool2, 192, 384, 1, 1, 'conv8_1'), training=True, name='bn8_1'))
        conv9 = relu(tf.layers.batch_normalization(conv2d(conv8, 96, 96, 3, 1, 'conv9'), training=True, name='bn9'))
        conv10 = relu(
            tf.layers.batch_normalization((conv2d(conv9, 96, 384, 1, 1, 'conv10')), training=True, name='bn10') + conv8_1)
        pool3 = pool(conv10)

        conv11 = relu(tf.layers.batch_normalization(conv2d(pool3, 384, 192, 1, 1, 'conv11'), training=True, name='bn11'))
        conv11_1 = tf.layers.batch_normalization(conv2d(pool3, 384, 768, 1, 1, 'conv11_1'), training=True, name='bn11_1')
        conv12 = relu(tf.layers.batch_normalization(conv2d(conv11, 192, 192, 3, 1, 'conv12'), training=True, name='bn12'))
        conv13 = relu(tf.layers.batch_normalization(conv2d(conv12, 192, 768, 1, 1, 'conv13'), training=True,
                                                    name='bn13') + conv11_1)



        fullconnetion1 = relu(
            tf.layers.batch_normalization(fullcon(conv13, 768, 32, 28, 1, 'full1'), training=True, name='bn20'))
        #fullconnetion2 = relu(
            #tf.layers.batch_normalization(fullcon(fullconnetion1, 768, 384, 1, 1, 'full2'), training=True, name='bn21'))
        fullconnetion3 = fullcon_b(fullconnetion1, 32, 2, 1, 1, 'full3')

        output=tf.squeeze(fullconnetion3,[1,2])
        return output
def lrelu(x):
    return tf.maximum(0.2*x,x)
def discriminator_s(input,reuse):
    with tf.variable_scope('dis',reuse=reuse):

        conv1 = lrelu(tf.layers.batch_normalization(conv3d(input, 1, 32, 3, 1, 'conv1'), training=True, name='bn1'))
        conv1_1 = lrelu(tf.layers.batch_normalization(conv3d(conv1, 32, 32, 3, 2, 'conv1_1'), training=True, name='bn1_1'))
        conv2 = lrelu(tf.layers.batch_normalization(conv3d(conv1_1, 32, 64, 3, 1, 'conv2'), training=True, name='bn2'))
        conv2_1 = lrelu(tf.layers.batch_normalization(conv3d(conv2, 64, 64, 3, 2, 'conv2_1'), training=True, name='bn2_1'))
        conv3 = lrelu(tf.layers.batch_normalization(conv3d(conv2_1, 64, 128, 3, 1, 'conv3'), training=True, name='bn3'))
        conv3_1 = lrelu(tf.layers.batch_normalization(conv3d(conv3, 128, 128, 3, 2, 'conv3_1'), training=True, name='bn3_1'))
        conv4 = lrelu(tf.layers.batch_normalization(conv3d(conv3_1, 128, 256, 3, 1, 'conv4'), training=True, name='bn4'))


        # fullconnetion1 = relu(
        #     tf.layers.batch_normalization(fullcon(conv5, 512, 32, 7, 1, 'full1'), training=True, name='bn20'))
        #fullconnetion2 = relu(
            #tf.layers.batch_normalization(fullcon(fullconnetion1, 768, 384, 1, 1, 'full2'), training=True, name='bn21'))=
        flatten = tf.reshape(conv4, (4,32768))
        dense =tf.layers.dense(flatten,64)
        output = tf.exp(tf.layers.dense(dense, 1))
        return output
def gloss_1(cosout,coslabel,dis_cosout):
    loss1=tf.reduce_mean(tf.square(cosout-coslabel))
    loss3=0.000000005*gdl3d(cosout,coslabel)
    #loss2=tf.reduce_mean(tf.square(dis_cosout - tf.constant(1.0,shape=[4, 1])))
    #loss2=tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant([[1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0]]), logits=dis_cosout))

    loss2=0.00002*tf.reduce_mean(tf.square(dis_cosout-tf.random_uniform(shape=[4,1],minval=0.99,maxval=1.0,dtype=tf.float32)))

    loss=loss2+loss1+loss3
    return loss,loss1,loss2,loss3
def gloss_0(cosout,coslabel,dis_cosout):
    loss1=tf.reduce_mean(tf.square(cosout-coslabel))
    loss3=0.000000005*gdl3d(cosout,coslabel)
    #loss2=tf.reduce_mean(tf.square(dis_cosout - tf.constant(1.0,shape=[4, 1])))
    #loss2=tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant([[1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0]]), logits=dis_cosout))

    loss2=0.00002*tf.reduce_mean(tf.square(dis_cosout-tf.random_uniform(shape=[4,1],minval=0.7,maxval=1.2,dtype=tf.float32)))

    loss=loss1
    return loss, loss1, loss2, loss3
def gloss_2(cosout,coslabel,dis_cosout):
    loss1=tf.reduce_mean(tf.square(cosout-coslabel))
    loss3=0.000000005*gdl3d(cosout,coslabel)
    #loss2=tf.reduce_mean(tf.square(dis_cosout - tf.constant(1.0,shape=[4, 1])))
    #loss2=tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant([[1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0]]), logits=dis_cosout))

    loss2=0.00002*tf.reduce_mean(tf.square(dis_cosout-tf.random_uniform(shape=[4,1],minval=0.7,maxval=1.2,dtype=tf.float32)))

    loss=loss2+loss1+loss3
    return loss,loss1,loss2,loss3
def gloss_3(cosout,coslabel,dis_cosout):
    loss1=tf.reduce_mean(tf.square(cosout-coslabel))
    loss3=0.000000005*gdl3d(cosout,coslabel)
    #loss2=tf.reduce_mean(tf.square(dis_cosout - tf.constant(1.0,shape=[4, 1])))
    #loss2=tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant([[1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0]]), logits=dis_cosout))

    loss2=0.00002*tf.reduce_mean(tf.square(dis_cosout-tf.random_uniform(shape=[4,1],minval=0.7,maxval=1.2,dtype=tf.float32)))

    loss=loss1+loss3
    return loss,loss1,loss2,loss3
def gloss_ref(cosout,refcosout,coslabel,dis_cosout):
    loss1=tf.reduce_mean(tf.square(cosout-coslabel))+tf.reduce_mean(tf.square(refcosout-coslabel))
    loss3=0.000000005*gdl3d(cosout,coslabel)+0.000000005*gdl3d(refcosout,coslabel)
    #loss2=tf.reduce_mean(tf.square(dis_cosout - tf.constant(1.0,shape=[4, 1])))
    #loss2=tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant([[1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0]]), logits=dis_cosout))

    loss2=0.00004*tf.reduce_mean(tf.square(dis_cosout-tf.random_uniform(shape=[4,1],minval=0.7,maxval=1.2,dtype=tf.float32)))

    loss=loss2+loss1+loss3
    return loss,loss1,loss2,loss3
def dloss(neg,pos):

    pos_loss = tf.reduce_mean(tf.square(pos - tf.random_uniform(shape=[4, 1], minval=0.7, maxval=1.2)))
    #pos_loss =tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant([[1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0]]), logits=pos))
    neg_loss = tf.reduce_mean(tf.square(neg - tf.random_uniform(shape=[4, 1], minval=0, maxval=0.3, dtype=tf.float32)))
    #neg_loss =tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant([[0.0, 1.0],[0.0, 1.0],[0.0, 1.0],[0.0, 1.0]]), logits=neg))
    d_loss = neg_loss + pos_loss
    return d_loss
def dloss1(neg,pos):

    pos_loss = tf.reduce_mean(tf.square(pos - tf.random_uniform(shape=[4, 1], minval=0.99, maxval=1.0)))
    #pos_loss =tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant([[1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0]]), logits=pos))
    neg_loss = tf.reduce_mean(tf.square(neg - tf.random_uniform(shape=[4, 1], minval=0, maxval=0.01, dtype=tf.float32)))
    #neg_loss =tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant([[0.0, 1.0],[0.0, 1.0],[0.0, 1.0],[0.0, 1.0]]), logits=neg))
    d_loss = neg_loss + pos_loss
    return d_loss
def gdl3d(gen_frames, gt_frames, alpha=1):
    """
    Calculates the sum of GDL losses between the predicted and ground truth frames.
    This is the 3d version.
    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each scale
    @param alpha: The power to which each gradient term is raised.
    @return: The GDL loss for 3d. Dong
    """
    # calculate the loss for each scale
    scale_losses = []

    # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
    pos = tf.constant(np.identity(1), dtype=tf.float32)
    neg = -1 * pos

    baseFilter = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]# 2x1x1x1
    filter_x = tf.expand_dims(baseFilter, 1)  # [-1, 1] # 2x1x1x1x1
    filter_y = tf.expand_dims(baseFilter, 0)  # [-1, 1] # 1x2x1x1x1
    filter_z = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1] # 1x2x1x1
    filter_z = tf.expand_dims(filter_z, 0) # [-1, 1] #1x1x2x1x1
    strides = [1, 1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'

    gen_dx = tf.abs(tf.nn.conv3d(gen_frames, filter_x, strides, padding=padding))
    gen_dy = tf.abs(tf.nn.conv3d(gen_frames, filter_y, strides, padding=padding))
    gen_dz = tf.abs(tf.nn.conv3d(gen_frames, filter_z, strides, padding=padding))
    gt_dx = tf.abs(tf.nn.conv3d(gt_frames, filter_x, strides, padding=padding))
    gt_dy = tf.abs(tf.nn.conv3d(gt_frames, filter_y, strides, padding=padding))
    gt_dz = tf.abs(tf.nn.conv3d(gt_frames, filter_z, strides, padding=padding))

    grad_diff_x = tf.abs(gt_dx - gen_dx)
    grad_diff_y = tf.abs(gt_dy - gen_dy)
    grad_diff_z = tf.abs(gt_dz - gen_dz)

    scale_losses.append(tf.reduce_sum((grad_diff_x ** alpha + grad_diff_y ** alpha + grad_diff_z ** alpha)))

    # condense into one tensor and avg
    return tf.reduce_mean(tf.stack(scale_losses))
def pre_cos_ssim(input1):
    input1[input1<-0.15]=-0.15
    input1[input1 > 0.2] = 0.2
    input2=(input1+0.15)/0.35
    # max1=input1.max()
    # min1=input1.min()
    # input2=(input1-min1)/(max1-min1)
    input3=input2*255
    return input3
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def instance_norm(x):
    epsilon = 1e-9

    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))


def load_magimage(path):
    image = sio.loadmat(path)
    image = image['magmap']
    #image = (image + 1) / 2
    # image = image - MEAN_VALUES

    # Resize the image for convnet input, there is no change but just
    # add an extra dimension.
    image = np.reshape(image, ((1,) + image.shape))
    image = image[:, :,:, :, np.newaxis]
    # Input to the VGG net expects the mean to be subtracted.
    image = image.astype('float32')
    #image=tf.convert_to_tensor(image)
    return image
def load_fieldimage(path):
    image = sio.loadmat(path)
    image = image['field']
    #image=(image+1)/2

    # image = image - MEAN_VALUES

    # Resize the image for convnet input, there is no change but just
    # add an extra dimension.
    image = np.reshape(image, ((1,) + image.shape))
    image = image[:, :,:, :, np.newaxis]
    # Input to the VGG net expects the mean to be subtracted.
    image = image.astype('float32')
    # image=tf.convert_to_tensor(image)
    return image
def load_cosmos(path):
    image = sio.loadmat(path)
    image = image['chi_mo']
    #image=(image+1)/2

    # image = image - MEAN_VALUES

    # Resize the image for convnet input, there is no change but just
    # add an extra dimension.
    image = np.reshape(image, ((1,) + image.shape))
    image = image[:, :, :,:, np.newaxis]
    # Input to the VGG net expects the mean to be subtracted.
    image = image.astype('float32')
    # image=tf.convert_to_tensor(image)
    return image
def load_tkd(path):
    image = sio.loadmat(path)
    image = image['tkd']
    #image=(image+1)/2

    # image = image - MEAN_VALUES

    # Resize the image for convnet input, there is no change but just
    # add an extra dimension.
    image = np.reshape(image, ((1,) + image.shape))
    image = image[:, :, :,:, np.newaxis]
    # Input to the VGG net expects the mean to be subtracted.
    image = image.astype('float32')
    # image=tf.convert_to_tensor(image)
    return image