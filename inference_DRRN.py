import tensorflow as tf

conv1_size, inp1_channel, out1_channel = 3, 1, 128

conv2_size, inp2_channel, out2_channel = 3, 128, 128
conv3_size, inp3_channel, out3_channel = 3, 128, 128

##
conv4_size, inp4_channel, out4_channel = 3, 128, 128

conv5_size, inp5_channel, out5_channel = 3, 128, 128
conv6_size, inp6_channel, out6_channel = 3, 128, 128

##
conv7_size, inp7_channel, out7_channel = 3, 128, 128

conv8_size, inp8_channel, out8_channel = 3, 128, 128
conv9_size, inp9_channel, out9_channel = 3, 128, 128

##
conv10_size, inp10_channel, out10_channel = 3, 128, 128

conv11_size, inp11_channel, out11_channel = 3, 128, 128
conv12_size, inp12_channel, out12_channel = 3, 128, 128

##
conv13_size, inp13_channel, out13_channel = 3, 128, 128

conv14_size, inp14_channel, out14_channel = 3, 128, 128
conv15_size, inp15_channel, out15_channel = 3, 128, 128

##
conv16_size, inp16_channel, out16_channel = 3, 128, 128

conv17_size, inp17_channel, out17_channel = 3, 128, 128
conv18_size, inp18_channel, out18_channel = 3, 128, 128

##
netop_conv_size, netop_inp_channel, netop_out_channel = 3, 128, 1

def int_w(shape,name):
    weights = tf.get_variable(
        name=name, shape=shape,dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
        # initializer=tf.contrib.layers.xavier_initializer()
    )
    return weights


def net(input):
    # input_tensor = tf.expand_dims(input, -1)
    ###### RB-1
    w1 = int_w(shape=[1, conv1_size, inp1_channel, out1_channel], name='W1')
    conv1 = tf.nn.conv2d(
        input, w1, strides=[1, 1, 1, 1], padding='SAME', name='conv1'
    )

    w2 = int_w(shape=[1, conv2_size, inp2_channel, out2_channel], name='W2')
    w3 = int_w(shape=[1, conv3_size, inp3_channel, out3_channel], name='W3')

    out1 = conv1
    for i in range(3):
        ##
        RB1_relu_part1 = tf.nn.relu(tf.layers.batch_normalization(out1,training=True))
        RB1_part1 = tf.nn.conv2d(
            RB1_relu_part1, w2, strides=[1, 1, 1, 1], padding='SAME', name='conv2'
        )
        ##
        RB1_relu_part2 = tf.nn.relu(tf.layers.batch_normalization(RB1_part1, training=True))
        RB1_part2 = tf.nn.conv2d(
            RB1_relu_part2, w3, strides=[1, 1, 1, 1], padding='SAME', name='conv3'
        )
        out1 = conv1 + RB1_part2
    ###### RB-2
    w4 = int_w(shape=[1, conv4_size, inp4_channel, out4_channel], name='W4')
    conv4 = tf.nn.conv2d(
        out1, w4, strides=[1, 1, 1, 1], padding='SAME', name='conv4'
    )
    w5 = int_w(shape=[1, conv5_size, inp5_channel, out5_channel], name='W5')
    w6 = int_w(shape=[1, conv6_size, inp6_channel, out6_channel], name='W6')

    out2 = conv4
    for i in range(3):
        RB2_relu_part1 = tf.nn.relu(tf.layers.batch_normalization(out2, training=True))
        RB2_part1 = tf.nn.conv2d(
            RB2_relu_part1, w5, strides=[1, 1, 1, 1], padding='SAME', name='conv5'
        )
        ##
        RB2_relu_part2 = tf.nn.relu(tf.layers.batch_normalization(RB2_part1, training=True))
        RB2_part2 = tf.nn.conv2d(
            RB2_relu_part2, w6, strides=[1, 1, 1, 1], padding='SAME', name='conv6'
        )
        out2 = conv4 + RB2_part2

    ###### RB-3
    w7 = int_w(shape=[1, conv7_size, inp7_channel, out7_channel], name='W7')
    conv7 = tf.nn.conv2d(
        out2, w7, strides=[1, 1, 1, 1], padding='SAME', name='conv7'
    )
    w8 = int_w(shape=[1, conv8_size, inp8_channel, out8_channel], name='W8')
    w9 = int_w(shape=[1, conv9_size, inp9_channel, out9_channel], name='W9')

    out3 = conv7
    for i in range(3):
        RB3_relu_part1 = tf.nn.relu(tf.layers.batch_normalization(out3, training=True))
        RB3_part1 = tf.nn.conv2d(
            RB3_relu_part1, w8, strides=[1, 1, 1, 1], padding='SAME', name='conv8'
        )
        ##
        RB3_relu_part2 = tf.nn.relu(tf.layers.batch_normalization(RB3_part1, training=True))
        RB3_part2 = tf.nn.conv2d(
            RB3_relu_part2, w9, strides=[1, 1, 1, 1], padding='SAME', name='conv9'
        )
        out3 = conv7 + RB3_part2

    ###### RB-4
    w10 = int_w(shape=[1, conv10_size, inp10_channel, out10_channel], name='W10')
    conv10 = tf.nn.conv2d(
        out3, w10, strides=[1, 1, 1, 1], padding='SAME', name='conv10'
    )
    w11 = int_w(shape=[1, conv11_size, inp11_channel, out11_channel], name='W11')
    w12 = int_w(shape=[1, conv12_size, inp12_channel, out12_channel], name='W12')

    out4 = conv10
    for i in range(3):
        RB4_relu_part1 = tf.nn.relu(tf.layers.batch_normalization(out4, training=True))
        RB4_part1 = tf.nn.conv2d(
            RB4_relu_part1, w11, strides=[1, 1, 1, 1], padding='SAME', name='conv11'
        )
        ##
        RB4_relu_part2 = tf.nn.relu(tf.layers.batch_normalization(RB4_part1, training=True))
        RB4_part2 = tf.nn.conv2d(
            RB4_relu_part2, w12, strides=[1, 1, 1, 1], padding='SAME', name='conv12'
        )
        out4 = conv10 + RB4_part2

    ###### RB-5
    w13 = int_w(shape=[1, conv13_size, inp13_channel, out13_channel], name='W13')
    conv13 = tf.nn.conv2d(
        out4, w13, strides=[1, 1, 1, 1], padding='SAME', name='conv13'
    )
    w14 = int_w(shape=[1, conv14_size, inp14_channel, out14_channel], name='W14')
    w15 = int_w(shape=[1, conv15_size, inp15_channel, out15_channel], name='W15')

    out5 = conv13
    for i in range(3):
        RB5_relu_part1 = tf.nn.relu(tf.layers.batch_normalization(out5, training=True))
        RB5_part1 = tf.nn.conv2d(
            RB5_relu_part1, w14, strides=[1, 1, 1, 1], padding='SAME', name='conv14'
        )
        ##
        RB5_relu_part2 = tf.nn.relu(tf.layers.batch_normalization(RB5_part1, training=True))
        RB5_part2 = tf.nn.conv2d(
            RB5_relu_part2, w15, strides=[1, 1, 1, 1], padding='SAME', name='conv15'
        )
        out5 = conv13 + RB5_part2

    ###### RB-6
    w16 = int_w(shape=[1, conv16_size, inp16_channel, out16_channel], name='W16')
    conv16 = tf.nn.conv2d(
        out5, w16, strides=[1, 1, 1, 1], padding='SAME', name='conv16'
    )
    w17 = int_w(shape=[1, conv17_size, inp17_channel, out17_channel], name='W17')
    w18 = int_w(shape=[1, conv18_size, inp18_channel, out18_channel], name='W18')

    out6 = conv16
    for i in range(3):
        RB6_relu_part1 = tf.nn.relu(tf.layers.batch_normalization(out6, training=True))
        RB6_part1 = tf.nn.conv2d(
            RB6_relu_part1, w17, strides=[1, 1, 1, 1], padding='SAME', name='conv16'
        )
        ##
        RB6_relu_part2 = tf.nn.relu(tf.layers.batch_normalization(RB6_part1, training=True))
        RB6_part2 = tf.nn.conv2d(
            RB6_relu_part2, w18, strides=[1, 1, 1, 1], padding='SAME', name='conv18'
        )
        out6 = conv16 + RB6_part2

    w_netop = int_w(shape=[1, netop_conv_size, netop_inp_channel, netop_out_channel], name='W_netop')
    conv_netop = tf.nn.conv2d(
        out6, w_netop, strides=[1, 1, 1, 1], padding='SAME', name='conv_netop'
    )
    net_out = conv_netop + input
    # net_out = tf.squeeze(net_out,[-1])

    return net_out