import tensorflow as tf
import math

EPS = 10e-5
# batch_size = 10
input_us = 512
FullConect1 = 512
FullConect2 = 512
FullConect3 = 512

conv1_size = 3
inp1_channel = 1
out1_channel = 64

conv2_size = 3
inp2_channel = 64
out2_channel = 64

conv3_size = 3
inp3_channel = 64
out3_channel = 128

conv4_size = 3
inp4_channel = 128
out4_channel = 128

conv5_size = 3
inp5_channel = 128
out5_channel = 256

conv6_size = 3
inp6_channel = 256
out6_channel = 256

conv7_size = 3
inp7_channel = 256
out7_channel = 512

conv8_size = 3
inp8_channel = 512
out8_channel = 512

conv9_size = 3
inp9_channel = 512
out9_channel = 1024

conv10_size = 3
inp10_channel = 1024
out10_channel = 1024

transconv1 = 3
up_out1_channel = 512
up_inp1_channel = 1024


conv12_size = 3
inp12_channel = 1024
out12_channel = 512

conv13_size = 3
inp13_channel = 512
out13_channel = 512

transconv2 = 3
up_out2_channel = 256
up_inp2_channel = 512

conv15_size = 3
inp15_channel = 512
out15_channel = 256

conv16_size = 3
inp16_channel = 256
out16_channel = 256

transconv3 = 3
up_out3_channel = 128
up_inp3_channel = 256


conv18_size = 3
inp18_channel = 256
out18_channel = 128

conv19_size = 3
inp19_channel = 128
out19_channel = 128

transconv4 = 1
up_out4_channel = 64
up_inp4_channel = 128

conv21_size = 3
inp21_channel = 128
out21_channel = 64

conv22_size = 3
inp22_channel = 64
out22_channel = 64

conv23_size = 3
inp23_channel = 64
out23_channel = 1

def batch_norm(x, is_training, eps=EPS, decay=0.9, affine=True, name='BatchNorm2d'):
    from tensorflow.python.training.moving_averages import assign_moving_average

    with tf.variable_scope(name):
        params_shape = x.shape[-1:]
        moving_mean = tf.get_variable(name='mean', shape=params_shape, initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_var = tf.get_variable(name='variance', shape=params_shape, initializer=tf.ones_initializer,
                                     trainable=False)

        def mean_var_with_update():
            mean_this_batch, variance_this_batch = tf.nn.moments(x, list(range(len(x.shape) - 1)), name='moments')
            with tf.control_dependencies([
                assign_moving_average(moving_mean, mean_this_batch, decay),
                assign_moving_average(moving_var, variance_this_batch, decay)
            ]):
                return tf.identity(mean_this_batch), tf.identity(variance_this_batch)

        mean, variance = tf.cond(is_training, mean_var_with_update, lambda: (moving_mean, moving_var))
        if affine:  # 如果要用beta和gamma进行放缩
            beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
            normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=beta, scale=gamma,
                                               variance_epsilon=eps)
        else:
            normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=None, scale=None,
                                               variance_epsilon=eps)
        return normed


def int_w(shape,name):
    weights = tf.get_variable(
        name=name, shape=shape,dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
        # initializer=tf.contrib.layers.xavier_initializer()
    )
    return weights


def net(input,batch_size,len):
    len_layer1 = len
    len_layer2 = math.ceil(len_layer1/2)
    len_layer3 = math.ceil(len_layer2/2)
    len_layer4 = math.ceil(len_layer3/2)
    #### fully-connected
        #fully1
    # W1_FlCon = tf.get_variable(
    #     "W1_FlCon", [input_us, FullConect1],
    #     initializer=tf.truncated_normal_initializer(stddev=0.1)
    #     # initializer=tf.contrib.layers.xavier_initializer()
    # )
    # b1_FlCon = tf.get_variable(
    #     "b1_FlCon", [FullConect1], initializer=tf.constant_initializer(0.0)
    # )
    # hideen1 = tf.nn.relu(tf.matmul(input, W1_FlCon) + b1_FlCon)
    #     # fully2
    # W2_FlCon = tf.get_variable(
    #     "W2_FlCon", [FullConect1, FullConect2],
    #     initializer=tf.truncated_normal_initializer(stddev=0.1)
    #     # initializer=tf.contrib.layers.xavier_initializer()
    # )
    # b2_FlCon = tf.get_variable(
    #     "b2_FlCon", [FullConect2], initializer=tf.constant_initializer(0.0)
    # )
    # hideen2 = tf.nn.relu(tf.matmul(hideen1, W2_FlCon) + b2_FlCon)
    #    # fully3
    # # W3_FlCon = tf.get_variable(
    # #     "W3_FlCon", [FullConect2, FullConect3],
    # #     initializer=tf.truncated_normal_initializer(stddev=0.1)
    # #     # initializer=tf.contrib.layers.xavier_initializer()
    # # )
    # # b3_FlCon = tf.get_variable(
    # #     "b3_FlCon", [FullConect3], initializer=tf.constant_initializer(0.0)
    # # )
    # # hideen3 = tf.nn.relu(tf.matmul(hideen2, W3_FlCon) + b3_FlCon)
    #
    # input_conv = tf.expand_dims(hideen2,1)
    # input_conv = tf.expand_dims(input_conv, -1)

    ### layer1
       #conv1
    w1 = int_w(shape=[1,conv1_size,inp1_channel,out1_channel],name='W1')
    conv1 = tf.nn.conv2d(
        input, w1, strides=[1, 1, 1, 1], padding='SAME',name='conv1'
    )
    # relu1 = tf.nn.relu(tf.layers.batch_normalization(conv1,training=True))
    relu1 = tf.nn.relu(conv1)
       #conv2
    w2 = int_w(shape=[1,conv2_size,inp2_channel,out2_channel],name='W2')
    conv2 = tf.nn.conv2d(
        relu1,w2,strides=[1, 1, 1, 1],padding='SAME',name='conv2'
    )
    # relu2 = tf.nn.relu(tf.layers.batch_normalization(conv2,training=True))
    relu2 = tf.nn.relu(conv2)
    pool1 = tf.nn.max_pool(
        relu2,ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],padding='SAME'
    )
    # result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=keep_prob)

    ### layer2
    # conv1
    w3 = int_w(shape=[1,conv3_size,inp3_channel,out3_channel],name='W3')
    conv3 = tf.nn.conv2d(
        pool1, w3, strides=[1, 1, 1, 1], padding='SAME',name='conv3'
    )
    # relu3 = tf.nn.relu(tf.layers.batch_normalization(conv3,training=True))
    relu3 = tf.nn.relu(conv3)
       #conv2
    w4 = int_w(shape=[1,conv4_size,inp4_channel,out4_channel],name='W4')
    conv4 = tf.nn.conv2d(
        relu3,w4,strides=[1, 1, 1, 1],padding='SAME',name='conv4'
    )
    # relu4 = tf.nn.relu(tf.layers.batch_normalization(conv4,training=True))
    relu4 = tf.nn.relu(conv4)
    pool2 = tf.nn.max_pool(
        relu4,ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],padding='SAME'
    )

    ### layer3
    # conv1
    w5 = int_w(shape=[1, conv5_size, inp5_channel, out5_channel], name='W5')
    conv5 = tf.nn.conv2d(
        pool2, w5, strides=[1, 1, 1, 1], padding='SAME', name='conv5'
    )
    # relu5 = tf.nn.relu(tf.layers.batch_normalization(conv5,training=True))
    relu5 = tf.nn.relu(conv5)

    # conv2
    w6 = int_w(shape=[1, conv6_size, inp6_channel, out6_channel], name='W6')
    conv6 = tf.nn.conv2d(
        relu5, w6, strides=[1, 1, 1, 1], padding='SAME', name='conv6'
    )
    # relu6 = tf.nn.relu(tf.layers.batch_normalization(conv6,training=True))
    relu6 = tf.nn.relu(conv6)

    pool3 = tf.nn.max_pool(
        relu6, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME'
    )

    ### layer4
    # conv1
    w7 = int_w(shape=[1, conv7_size, inp7_channel, out7_channel], name='W7')
    conv7 = tf.nn.conv2d(
        pool3, w7, strides=[1, 1, 1, 1], padding='SAME', name='conv7'
    )
    # relu7 = tf.nn.relu(tf.layers.batch_normalization(conv7,training=True))
    relu7 = tf.nn.relu(conv7)

    # conv2
    w8 = int_w(shape=[1, conv8_size, inp8_channel, out8_channel], name='W8')
    conv8 = tf.nn.conv2d(
        relu7, w8, strides=[1, 1, 1, 1], padding='SAME', name='conv8'
    )
    # relu8 = tf.nn.relu(tf.layers.batch_normalization(conv8,training=True))
    relu8 = tf.nn.relu(conv8)


    pool4 = tf.nn.max_pool(
        relu8, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME'
    )
    ### layer5(bottom)
    # conv1
    w9 = int_w(shape=[1, conv9_size, inp9_channel, out9_channel], name='W9')
    conv9 = tf.nn.conv2d(
        pool4, w9, strides=[1, 1, 1, 1], padding='SAME', name='conv9'
    )
    # relu9 = tf.nn.relu(tf.layers.batch_normalization(conv9,training=True))
    relu9 = tf.nn.relu(conv9)

    # conv2
    w10 = int_w(shape=[1, conv10_size, inp10_channel, out10_channel], name='W10')
    conv10 = tf.nn.conv2d(
        relu9, w10, strides=[1, 1, 1, 1], padding='SAME', name='conv10'
    )
    # relu10 = tf.nn.relu(tf.layers.batch_normalization(conv10,training=True))
    relu10 = tf.nn.relu(conv10)


    # up sample
    w11 = int_w(shape=[1, transconv1, up_out1_channel, up_inp1_channel], name='w11')
    conv11 = tf.nn.conv2d_transpose(
        relu10, w11,output_shape=[batch_size, 1, len_layer4, 512],strides=[1, 1, 2, 1], padding='SAME'
    )
    # relu11 = tf.nn.relu(tf.layers.batch_normalization(conv11,training=True))
    relu11 = tf.nn.relu(conv11)


    ### layer6
    # conv1
    concat1 = tf.concat([relu8,relu11],-1)

    # conv1
    w12 = int_w(shape=[1, conv12_size, inp12_channel, out12_channel], name='W12')
    conv12 = tf.nn.conv2d(
        concat1, w12, strides=[1, 1, 1, 1], padding='SAME', name='conv12'
    )
    # relu12 = tf.nn.relu(tf.layers.batch_normalization(conv12,training=True))
    relu12 = tf.nn.relu(conv12)

    # conv2
    w13 = int_w(shape=[1, conv13_size, inp13_channel, out13_channel], name='W13')
    conv13 = tf.nn.conv2d(
        relu12, w13, strides=[1, 1, 1, 1], padding='SAME', name='conv113'
    )
    # relu13 = tf.nn.relu(tf.layers.batch_normalization(conv13,training=True))
    relu13 = tf.nn.relu(conv13)


    # up sample
    w14 = int_w(shape=[1, transconv2, up_out2_channel, up_inp2_channel], name='w14')
    conv14 = tf.nn.conv2d_transpose(
        relu13, w14, output_shape=[batch_size, 1, len_layer3, 256], strides=[1, 1, 2, 1], padding='SAME'
    )
    # relu14 = tf.nn.relu(tf.layers.batch_normalization(conv14,training=True))
    relu14 = tf.nn.relu(conv14)


    ### layer7
    # conv1
    concat2 = tf.concat([relu6, relu14], -1)

    # conv1
    w15 = int_w(shape=[1, conv15_size, inp15_channel, out15_channel], name='W15')
    conv15 = tf.nn.conv2d(
        concat2, w15, strides=[1, 1, 1, 1], padding='SAME', name='conv15'
    )
    # relu15 = tf.nn.relu(tf.layers.batch_normalization(conv15,training=True))
    relu15 = tf.nn.relu(conv15)

    # conv2
    w16 = int_w(shape=[1, conv16_size, inp16_channel, out16_channel], name='W16')
    conv16 = tf.nn.conv2d(
        relu15, w16, strides=[1, 1, 1, 1], padding='SAME', name='conv16'
    )
    # relu16 = tf.nn.relu(tf.layers.batch_normalization(conv16,training=True))
    relu16 = tf.nn.relu(conv16)


    # up sample
    w17 = int_w(shape=[1, transconv3, up_out3_channel, up_inp3_channel], name='w17')
    conv17 = tf.nn.conv2d_transpose(
        relu16, w17, output_shape=[batch_size, 1, len_layer2, 128], strides=[1, 1, 2, 1], padding='SAME'
    )
    # relu17 = tf.nn.relu(tf.layers.batch_normalization(conv17,training=True))
    relu17 = tf.nn.relu(conv17)


    ### layer8
    # conv1
    concat3 = tf.concat([relu4, relu17], -1)

    # conv1
    w18 = int_w(shape=[1, conv18_size, inp18_channel, out18_channel], name='W18')
    conv18 = tf.nn.conv2d(
        concat3, w18, strides=[1, 1, 1, 1], padding='SAME', name='conv18'
    )
    # relu18 = tf.nn.relu(tf.layers.batch_normalization(conv18,training=True))
    relu18 = tf.nn.relu(conv18)

    # conv2
    w19 = int_w(shape=[1, conv19_size, inp19_channel, out19_channel], name='W19')
    conv19 = tf.nn.conv2d(
        relu18, w19, strides=[1, 1, 1, 1], padding='SAME', name='conv19'
    )
    # relu19 = tf.nn.relu(tf.layers.batch_normalization(conv19,training=True))
    relu19 = tf.nn.relu(conv19)


    # up sample
    w20 = int_w(shape=[1, transconv4, up_out4_channel, up_inp4_channel], name='w20')
    conv20 = tf.nn.conv2d_transpose(
        relu19, w20, output_shape=[batch_size, 1, len_layer1, 64], strides=[1, 1, 2, 1], padding='SAME'
    )
    # relu20 = tf.nn.relu(tf.layers.batch_normalization(conv20,training=True))
    relu20 = tf.nn.relu(conv20)

    ### layer9
    # conv1
    concat4 = tf.concat([relu2, relu20], -1)

    # conv1
    w21 = int_w(shape=[1, conv21_size, inp21_channel, out21_channel], name='W21')
    conv21 = tf.nn.conv2d(
        concat4, w21, strides=[1, 1, 1, 1], padding='SAME', name='conv21'
    )
    # relu21 = tf.nn.relu(tf.layers.batch_normalization(conv21,training=True))
    relu21 = tf.nn.relu(conv21)

    # conv2
    w22 = int_w(shape=[1, conv22_size, inp22_channel, out22_channel], name='W22')
    conv22 = tf.nn.conv2d(
        relu21, w22, strides=[1, 1, 1, 1], padding='SAME', name='conv22'
    )
    # relu22 = tf.nn.relu(tf.layers.batch_normalization(conv22,training=True))
    relu22 = tf.nn.relu(conv22)

    # conv3
    w23 = int_w(shape=[1, conv23_size, inp23_channel, out23_channel], name='W23')
    conv23 = tf.nn.conv2d(
        relu22, w23, strides=[1, 1, 1, 1], padding='SAME', name='conv23'
    )
    net_out = conv23
    return net_out

