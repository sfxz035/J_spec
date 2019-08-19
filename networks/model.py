from networks.ops import *
import tensorflow as tf
import numpy as np
#paramaters
FILTER_DIM = 64
OUTPUT_C = 1
#deep 5
def inference(input,is_training=True,reuse = False,args = None,name='UNet'):
    with tf.variable_scope(name, reuse=reuse):
        ### NonLocal替代Dense
        L1_Dense = NonLocal(input,2,name='Dense1')
        L2_Dense = NonLocal(L1_Dense,2,name='Dense2')
        ##Dense
        # L1_Dense = ReLU(tf.layers.dense(tf.layers.flatten(input),units=256),name='ReLU_D1')
        # L2_Dense = ReLU(tf.layers.dense(L1_Dense,units=256),name='ReLU_D1')
        # L2_Dense = tf.expand_dims(L2_Dense,1)
        # L2_Dense = tf.expand_dims(L2_Dense,-1)

        L1_conv = conv_relu(L2_Dense,args.SPFILTER_DIM,k_h=1,k_w=7,name='conv_1')
        L1_1 = ReLU(conv_bn(L1_conv, FILTER_DIM, k_h=1,is_train=is_training, name='Conv2d_1_1'),name='ReLU_1_1')
        L1_2 = ReLU(conv_bn(L1_1, FILTER_DIM, k_h=1,is_train=is_training, name='Conv2d_1_2'),name='ReLU_1_2')
        L2_1 = tf.nn.max_pool(L1_2, [1, 1, 2, 1], [1, 1, 2, 1], padding = 'SAME',name = 'MaxPooling1')  ##

        L2_2 = ReLU(conv_bn(L2_1, FILTER_DIM*2, k_h=1, is_train=is_training,name='Conv2d_2_1'),name='ReLU_2_1')
        L2_3 = ReLU(conv_bn(L2_2, FILTER_DIM*2, k_h=1, is_train=is_training,name='Conv2d_2_2'),name='ReLU_2_2')
        L3_1 = tf.nn.max_pool(L2_3, [1, 1, 2, 1], [1, 1, 2, 1], padding = 'SAME',name = 'MaxPooling2')    ##

        L3_2 = ReLU(conv_bn(L3_1, FILTER_DIM*4, k_h=1, is_train=is_training,name='Conv2d_3_1'),name='ReLU_3_1')
        L3_3 = ReLU(conv_bn(L3_2, FILTER_DIM*4, k_h=1, is_train=is_training,name='Conv2d_3_2'),name='ReLU_3_2')
        L4_1 = tf.nn.max_pool(L3_3, [1, 1, 2, 1], [1, 1, 2, 1], padding = 'SAME',name = 'MaxPooling3')    ##

        L4_2 = ReLU(conv_bn(L4_1, FILTER_DIM*8, k_h=1, is_train=is_training,name='Conv2d_4_1'),name='ReLU_4_1')
        L4_3 = ReLU(conv_bn(L4_2, FILTER_DIM*8, k_h=1, is_train=is_training,name='Conv2d_4_2'),name='ReLU_4_2')
        L5_1 = tf.nn.max_pool(L4_3, [1, 1, 2, 1], [1, 1, 2, 1], padding = 'SAME',name = 'MaxPooling4')  ##

        L5_2 = ReLU(conv_bn(L5_1, FILTER_DIM*16, k_h=1, is_train=is_training,name='Conv2d_5_1'),name='ReLU_5_1')
        L5_3 = ReLU(conv_bn(L5_2, FILTER_DIM*16, k_h=1, is_train=is_training,name='Conv2d_5_2'),name='ReLU_5_2')

        L4_U1 = ReLU(Deconv2d_bn(L5_3, L4_3.get_shape().as_list(),k_h = 1, strides =[1, 1, 2, 1],is_train=is_training,name = 'Deconv2d4'),name='DeReLU4')
        L4_U1 = tf.concat((L4_3, L4_U1), -1)
        L4_U2 = ReLU(conv_bn(L4_U1, FILTER_DIM * 8, k_h=1, is_train=is_training,name='Conv2d_4_u1'),name='ReLU_4_u1')
        L4_U3 = ReLU(conv_bn(L4_U2, FILTER_DIM * 8, k_h=1, is_train=is_training,name='Conv2d_4_u2'),name='ReLU_4_u2')

        L3_U1 = ReLU(Deconv2d_bn(L4_U3,L3_3.get_shape().as_list(),k_h = 1, strides =[1, 1, 2, 1],is_train=is_training,name = 'Deconv2d3'),name = 'DeReLU3')
        L3_U1 = tf.concat((L3_3, L3_U1), -1)
        L3_U2 = ReLU(conv_bn(L3_U1, FILTER_DIM*4, k_h=1,is_train=is_training, name='Conv2d_3_u1'), name='ReLU_3_u1')
        L3_U3 = ReLU(conv_bn(L3_U2, FILTER_DIM*4, k_h=1,is_train=is_training, name='Conv2d_3_u2'), name='ReLU_3_u2')

        L2_U1 = ReLU(Deconv2d_bn(L3_U3,L2_3.get_shape().as_list(), k_h = 1, strides =[1, 1, 2, 1],is_train=is_training,name = 'Deconv2d2'),name='DeReLU2')
        L2_U1 = tf.concat((L2_3, L2_U1), -1)
        L2_U2 = ReLU(conv_bn(L2_U1, FILTER_DIM*2, k_h=1, is_train=is_training,name='Conv2d_2_u1'),name='ReLU_2_u1')
        L2_U3 = ReLU(conv_bn(L2_U2, FILTER_DIM*2, k_h=1, is_train=is_training,name='Conv2d_2_u2'),name='ReLU_2_u2')

        L1_U1 = ReLU(Deconv2d_bn(L2_U3, L1_2.get_shape().as_list(),k_h = 1, strides =[1, 1, 2, 1],is_train=is_training,name='Deconv2d1'),name='DeReLU1')
        L1_U1 = tf.concat((L1_2, L1_U1), 3)
        L1_U2 = ReLU(conv_bn(L1_U1, FILTER_DIM, k_h=1, is_train=is_training,name='Conv1d_1_u1'),name='ReLU_1_u1')
        L1_U3 = ReLU(conv_bn(L1_U2, FILTER_DIM, k_h=1, is_train=is_training,name='Conv1d_1_u2'),name='ReLU_1_u2')

        conv1 = ReLU(conv_bn(L1_U3, 32, k_h=1, is_train=is_training,name='Conv2d_1'),name='ReLU_1')
        out = conv_b(conv1, OUTPUT_C,name='Conv1d_out')
        out = out + L2_Dense
        return out
def ResNet(input,is_training=True,reuse = False,args = None,name='ResNet'):
    with tf.variable_scope(name, reuse=reuse):
        L1_Dense = ReLU(tf.layers.dense(tf.layers.flatten(input),units=256),name='ReLU_D1')
        L2_Dense = ReLU(tf.layers.dense(L1_Dense,units=256),name='ReLU_D1')
        L2_Dense = tf.expand_dims(L2_Dense,1)
        L2_Dense = tf.expand_dims(L2_Dense,-1)
        L1_conv = conv_relu(L2_Dense,args.SPFILTER_DIM,k_h=1,k_w=7,name='conv_1')
        x = L1_conv
        for i in range(args.nubBlocks):
            x = SE_resblock(x, args.SPFILTER_DIM, is_training=is_training, name='Block_' + str(i))
        conv_out1 = conv_relu(x,32,k_h=1,name='conv_out1')
        conv_out2 = conv_b(conv_out1,1,k_h=1,name='conv_out2')
        output = conv_out2+L2_Dense
        return output

def SE_resblock(input,feature_size=64,kernel_size=[1,3],strides=[1,1,1,1],is_training=True,name='resBlock'):
    with tf.variable_scope(name):
        conv1 = ReLU(conv_bn(input, feature_size, kernel_size[0], kernel_size[1], strides, is_train=is_training,name='Resblock_conv1'),)
        conv2 = conv_bn(conv1, feature_size, kernel_size[0], kernel_size[1], strides,is_train=is_training, name='Resblock_conv2')
        conv2 = SE_block(conv2,ratio=8,name='SE_block')
        out = ReLU(conv2 + input)
        return out

def resBlock(input,feature_size=64,kernel_size=[1,3],strides=[1,1,1,1],is_training=True,name='resBlock'):
    with tf.variable_scope(name):
        conv1 = ReLU(conv_bn(input,feature_size,kernel_size[0],kernel_size[1],strides=strides,is_train=is_training,name='Resblock_conv1'),name='Resblock_ReLU1')
        conv2 = conv_bn(conv1,feature_size,kernel_size[0],kernel_size[1],strides=strides,is_train=is_training,name='Resblock_conv2')
        output = conv2 + input
        return output








