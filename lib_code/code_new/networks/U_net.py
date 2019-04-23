from networks.ops import *
import tensorflow as tf
import numpy as np
#paramaters
flags = tf.flags
flags.DEFINE_integer('MAX_INTER', 200000, 'The number of trainning steps')
flags.DEFINE_integer('MAX_TO_KEEP', 10, 'The max number of model to save')
flags.DEFINE_integer('BATCH_SIZE', 8, 'The size of batch images [16]')
flags.DEFINE_float('BETA', 1e-6, 'TV Optimizer [8e-2]')
flags.DEFINE_integer(
    'STEP', None,
    'Which checkpoint should be load, None for final step of checkpoint [None]'
)
flags.DEFINE_float('LR', 1e-4, 'Learning rate of for Optimizer [1e-4]')
flags.DEFINE_integer('NUM_GPUS', 1, 'The number of GPU to use [1]')
flags.DEFINE_boolean('IS_TRAIN', True, 'True for train, else test. [True]')
flags.DEFINE_integer(
    'FILTER_DIM', 64,
    'The number of feature maps in all layers. [64]'
)
flags.DEFINE_boolean(
    'LOAD_MODEL', True,
    'True for load checkpoint and continue training. [True]'
)
flags.DEFINE_string(
    'MODEL_DIR', 'UNet_g_12.26',
    'If LOAD_MODEL, provide the MODEL_DIR. [./model/baseline/]'
)
flags.DEFINE_string(
    'DATA_DIR', '/data1/sf/J_spec/data/data_J_0fft/train',
    'the data set direction'
)
FLAGS = flags.FLAGS


#deep 5
def inference(images, reuse = False, name='UNet'):
    with tf.variable_scope(name, reuse=reuse):
        L1_1 = conv_relu(images, FLAGS.FILTER_DIM, k_h=1, name='Conv1d_1_1')
        L1_2 = conv_relu(L1_1, FLAGS.FILTER_DIM, k_h=1, name='Conv1d_1_2')
        L2_1 = tf.nn.max_pool(L1_2, [1, 1, 2, 1], [1, 1, 2, 1], padding = 'SAME',name = 'MaxPooling1')  ##

        L2_2 = conv_relu(L2_1, FLAGS.FILTER_DIM*2, k_h=1, name='Conv1d_2_1')
        L2_3 = conv_relu(L2_2, FLAGS.FILTER_DIM*2, k_h=1, name='Conv1d_2_2')
        L3_1 = tf.nn.max_pool(L2_3, [1, 1, 2, 1], [1, 1, 2, 1], padding = 'SAME',name = 'MaxPooling2')    ##

        L3_2 = conv_relu(L3_1, FLAGS.FILTER_DIM*4, k_h=1, name='Conv1d_3_1')
        L3_3 = conv_relu(L3_2, FLAGS.FILTER_DIM*4, k_h=1, name='Conv1d_3_2')
        L4_1 = tf.nn.max_pool(L3_3, [1, 1, 2, 1], [1, 1, 2, 1], padding = 'SAME',name = 'MaxPooling3')    ##

        L4_2 = conv_relu(L4_1, FLAGS.FILTER_DIM*8, k_h=1, name='Conv1d_4_1')
        L4_3 = conv_relu(L4_2, FLAGS.FILTER_DIM*8, k_h=1, name='Conv1d_4_2')
        L5_1 = tf.nn.max_pool(L4_3, [1, 1, 2, 1], [1, 1, 2, 1], padding = 'SAME',name = 'MaxPooling4')  ##

        L5_2 = conv_relu(L5_1, FLAGS.FILTER_DIM*16, k_h=1, name='Conv1d_5_1')
        L5_3 = conv_relu(L5_2, FLAGS.FILTER_DIM*16, k_h=1, name='Conv1d_5_2')

        L4_U1 = Deconv2d(L5_3, L4_3.get_shape(), k_h = 1, k_w = 3, strides =[1, 1, 2, 1],name = 'Deconv1d4')
        L4_U1 = tf.concat((L4_3, L4_U1), 3)
        L4_U2 = conv_relu(L4_U1, FLAGS.FILTER_DIM * 8, k_h=1, name='Conv1d_4_u1')
        L4_U3 = conv_relu(L4_U2, FLAGS.FILTER_DIM * 8, k_h=1, name='Conv1d_4_u2')

        L3_U1 = Deconv2d(L4_U3,L3_3.get_shape(), k_h = 1, k_w = 3, strides =[1, 1, 2, 1],name = 'Deconv1d3')
        L3_U1 = tf.concat((L3_3, L3_U1), 3)
        L3_U2 = conv_relu(L3_U1, FLAGS.FILTER_DIM*4, k_h=1, name='Conv1d_3_u1')
        L3_U3 = conv_relu(L3_U2, FLAGS.FILTER_DIM*4, k_h=1, name='Conv1d_3_u2')

        L2_U1 = Deconv2d(L3_U3,L2_3.get_shape(), k_h = 1, k_w = 3, strides =[1, 1, 2, 1],name = 'Deconv1d2')
        L2_U1 = tf.concat((L2_3, L2_U1), 3)
        L2_U2 = conv_relu(L2_U1, FLAGS.FILTER_DIM*2, k_h=1, name='Conv1d_2_u1')
        L2_U3 = conv_relu(L2_U2, FLAGS.FILTER_DIM*2, k_h=1, name='Conv1d_2_u2')

        L1_U1 = Deconv2d(L2_U3, L1_2.get_shape(), k_h=1, k_w=3, strides=[1, 1, 2, 1], name='Deconv1d1')
        L1_U1 = tf.concat((L1_2, L1_U1), 3)
        L1_U2 = conv_relu(L1_U1, FLAGS.FILTER_DIM*1, k_h=1, name='Conv1d_1_u1')
        L1_U3 = conv_relu(L1_U2, FLAGS.FILTER_DIM*1, k_h=1, name='Conv1d_1_u2')

        out = conv(L1_U3, FLAGS.OUTPUT_C,name='Conv1d_out')

    # variables = tf.contrib.framework.get_variables(name)

    return out

def losses(output, labels, name = 'losses'):
    with tf.name_scope(name):
        loss = tf.reduce_mean(tf.square(output - labels))
        return loss








