# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops


def weight_variable(shape,name=None,trainable=True, decay_mult = 0.0):
    weights = tf.get_variable(
        name, shape, tf.float32, trainable=trainable,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
        # initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
        # regularizer=tf.contrib.layers.l2_regularizer(decay_mult)
    )
    return weights

def bias_variable(shape,name=None, bias_start = 0.0, trainable = True, decay_mult = 0.0):
    bais = tf.get_variable(
        name, shape, tf.float32, trainable = trainable,
        initializer = tf.constant_initializer(bias_start, dtype = tf.float32)
        # regularizer = tf.contrib.layers.l2_regularizer(decay_mult)
    )
    return bais

def conv_bn(inpt ,output_dim, k_h = 3, k_w = 3, strides = [1, 1, 1, 1], is_train = True, name='Conv2d'):
    with tf.variable_scope(name):
        filter_ = weight_variable([k_h,k_w,inpt.get_shape()[-1],output_dim],name='weights')
        conv = tf.nn.conv2d(inpt, filter=filter_, strides=strides, padding="SAME")
        batch_norm = tf.layers.batch_normalization(conv, training=is_train) ###由contrib换成layers
    return batch_norm

def BatchNorm(
        value, is_train = True, name = 'BatchNorm',
        epsilon = 1e-5, momentum = 0.9
    ):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(
            value,
            decay = momentum,
            # updates_collections = tf.GraphKeys.UPDATE_OPS,
            # updates_collections = None,
            epsilon = epsilon,
            scale = True,
            is_training = is_train,
            scope = name
        )

def conv_relu(inpt, output_dim, k_h = 3, k_w = 3, strides = [1, 1, 1, 1],name='Conv2d'):
    with tf.variable_scope(name):
        filter_ = weight_variable([k_h, k_w, inpt.get_shape()[-1], output_dim],name='weights')
        conv = tf.nn.conv2d(inpt, filter=filter_, strides=strides, padding="SAME")
        biases = bias_variable(output_dim,name='biases')
        pre_relu = tf.nn.bias_add(conv, biases)
        out = tf.nn.relu(pre_relu)
        return out


def conv_b(inpt, output_dim, k_h = 3, k_w = 3, strides = [1, 1, 1, 1],name='Conv2d'):
    with tf.variable_scope(name):
        filter_ = weight_variable([k_h, k_w, inpt.get_shape()[-1], output_dim],name='weights')
        conv = tf.nn.conv2d(inpt, filter=filter_, strides=strides, padding="SAME")
        biases = bias_variable(output_dim,name='biases')
        out = tf.nn.bias_add(conv, biases)
    return out


def ReLU(value, name = 'ReLU'):
    with tf.variable_scope(name):
        return tf.nn.relu(value)

def Deconv2d(
        value, output_shape, k_h = 3, k_w = 3, strides =[1, 2, 2, 1],
        name = 'Deconv2d', with_w = False
    ):
    with tf.variable_scope(name):
        weights = weight_variable(
            name='weights',
            shape=[k_h, k_w, output_shape[-1], value.get_shape()[-1]],
            decay_mult = 1.0
        )
        deconv = tf.nn.conv2d_transpose(
            value, weights, output_shape, strides = strides
        )
        biases = bias_variable(name='biases', shape=[output_shape[-1]])
        deconv = tf.nn.bias_add(deconv, biases)
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, weights, biases
        else:
            return deconv
def Deconv2d_bn(
        value, output_shape, k_h = 3, k_w = 3, strides =[1, 2, 2, 1],
        is_train=True, name = 'Deconv2d', with_w = False
    ):
    with tf.variable_scope(name):
        weights = weight_variable(
            name='weights',
            shape=[k_h, k_w, output_shape[-1], value.get_shape()[-1]],
            decay_mult = 1.0
        )
        deconv = tf.nn.conv2d_transpose(
            value, weights, output_shape, strides = strides
        )
        batch_norm = tf.layers.batch_normalization(deconv, training=is_train) ###由contrib换成layers
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return batch_norm, weights
        else:
            return batch_norm
def SE_block(input_x, ratio=16, name='SE_block'):
    with tf.variable_scope(name):
        input_shape = input_x.get_shape().as_list()
        squeeze = tf.nn.avg_pool(input_x,[1,input_shape[1],input_shape[2],1],[1,input_shape[1],input_shape[2],1],padding='SAME')
        F1 = ReLU(Fully_connected(squeeze,units=input_shape[3]/ratio,name='Fc1'),name='ReLU_Fc1')
        F2 = Fully_connected(F1,units=input_shape[3],name='Fc2')
        excitation = tf.nn.sigmoid(F2)
        excitation = tf.reshape(excitation, [-1,1,1,input_shape[-1]])
        scale = input_x * excitation
        return scale
def Fully_connected(x, units, name='fully_connected') :
    with tf.name_scope(name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)


def NonLocal(net,depth,embed=True,softmax=True,name='NonLocal'):
    with tf.variable_scope(name):
        if embed:
            a = conv_b(net,depth,1,1,name='embA')
            b = conv_b(net,depth,1,1,name='embB')
        else:
            a, b = net, net
        g_orig = g = conv_b(net, depth, 1,1, name='g')
        # Flatten from (B,H,W,C) to (B,HW,C) or similar
        a_flat = tf.reshape(a, [tf.shape(a)[0], -1, tf.shape(a)[-1]])
        b_flat = tf.reshape(b, [tf.shape(b)[0], -1, tf.shape(b)[-1]])
        g_flat = tf.reshape(g, [tf.shape(g)[0], -1, tf.shape(g)[-1]])
        a_flat.set_shape([a.shape[0], a.shape[1] * a.shape[2] if None not in a.shape[1:3] else None, a.shape[-1]])
        b_flat.set_shape([b.shape[0], b.shape[1] * b.shape[2] if None not in b.shape[1:3] else None, b.shape[-1]])
        g_flat.set_shape([g.shape[0], g.shape[1] * g.shape[2] if None not in g.shape[1:3] else None, g.shape[-1]])
        # Compute f(a, b) -> (B,HW,HW)
        f = tf.matmul(a_flat, tf.transpose(b_flat, [0, 2, 1]))
        if softmax:
            f = tf.nn.softmax(f)
        else:
            f = f / tf.cast(tf.shape(f)[-1], tf.float32)
        # Compute f * g ("self-attention") -> (B,HW,C)
        fg = tf.matmul(f, g_flat)
        # Expand and fix the static shapes TF lost track of.
        fg = tf.reshape(fg, tf.shape(g_orig))
        return fg
def inception(net,depth,is_train = True,name='inception'):
    with tf.variable_scope(name):
        conv_1 = ReLU(conv_bn(net,int(depth/4),1,1,is_train=is_train,name='conv1x1'),name='ReLU_1x1')
        conv_2_1 = conv_relu(net,int(depth/4),1,1,name='conv1x1_3')
        conv_2_2 = ReLU(conv_bn(conv_2_1,int(depth/4),1,3,is_train=is_train,name='conv1x3'),name='ReLU_1x3')
        conv_3_1 = conv_relu(net,int(depth/4),1,1,name='conv1x1_5')
        conv_3_2 = ReLU(conv_bn(conv_3_1,int(depth/4),1,5,is_train=is_train,name='conv1x5'),name='ReLU_1x5')
        max_pool = tf.nn.max_pool(net, [1, 1, 3, 1], [1, 1, 1, 1], padding='SAME', name='MaxPooling1')  ##
        max_pool_conv = ReLU(conv_bn(max_pool,int(depth/4),1,1,is_train=is_train,name='pool_conv1x1'),name='ReLU_pool_conv')

        res = tf.concat((conv_1,conv_2_2,conv_3_2,max_pool_conv),-1)

        return res
