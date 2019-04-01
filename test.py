import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import dataset
import lib_loss
import inference_unet
import inference_DRRN
import scipy.io as sio
import inference_unet_0norm
import inference_unet_noise
import inference_unet_Falsenorm
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.InteractiveSession(config = config)



learning_rate = 0.0001
batch_size = 1

def train_test():
    trainfile_dir = '/data1/sf/J_spec/data/data_J_fft001_noise/data_train/train_Jfft_1'
    testfile_dir = '/data1/sf/J_spec/data/data_J_fft001_noise/data_test/test_Jfft_1'
    index = 1
    data_train = sio.loadmat(trainfile_dir)
    train_input = data_train.get('X_J_fft_com')
    length = train_input.shape[1]
    # train_input = np.expand_dims(train_input,-1)
    train_input = np.expand_dims(train_input,0)
    train_label = data_train.get('X_ps')
    train_label = np.expand_dims(train_label,-1)
    train_label = np.expand_dims(train_label,0)

    data_test = sio.loadmat(testfile_dir)
    test_input = data_test.get('X_J_fft_com')
    test_input = np.expand_dims(test_input,0)
    # test_input = np.expand_dims(test_input,-1)
    test_label = data_test.get('X_ps')
    test_label = np.expand_dims(test_label,-1)
    test_label = np.expand_dims(test_label,0)

    # x_train, y_train = dataset.get_data(trainfile_dir, 'X_J_fft_com', 'X_ps')
    # length = x_train.shape[2]
    # # x_train = np.expand_dims(x_train, -1)
    # y_train = np.expand_dims(y_train, -1)
    # x_test, y_test = dataset.get_data(testfile_dir, 'X_J_fft_com', 'X_ps')
    # # x_test = np.expand_dims(x_test, -1)
    # y_test = np.expand_dims(y_test, -1)
    # train_input = x_train[0 : batch_size]
    # train_label = y_train[0 : batch_size]
    # test_input = x_test[0 : batch_size]
    # test_label = y_test[0 : batch_size]

    x = tf.placeholder(tf.float32,shape = [batch_size,None,None, 2])
    y_ = tf.placeholder(tf.float32,shape = [batch_size,None,None,1])

    # y = inference_unet.net(x,batch,length)
    y = inference_unet_noise.net(x,len=length)

    loss = tf.reduce_mean(tf.square(y - y_))
    # loss = lib_loss.huber_loss(y, y_, delta=1.0)

    variables_to_restore = []
    for v in tf.global_variables():
        variables_to_restore.append(v)
    saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V1, max_to_keep=None)
    # tf.global_variables_initializer().run()
    saver.restore(sess, "F:/PycharmProjects/J_spec/lib_savenet/save_unet_Jfft001_noise/conv_unet1049999.ckpt-done")
    output = sess.run(y, feed_dict={x:train_input})
    output = np.squeeze(output)
    loss_train = sess.run(loss, feed_dict={x: train_input, y_: train_label})
    print('loss_train: %g' % (loss_train))
    output2 = sess.run(y, feed_dict={x: test_input})
    output2 = np.squeeze(output2)
    loss_test = sess.run(loss, feed_dict={x: test_input,
                                          y_: test_label})
    print('loss_test: %g' % (loss_test))
    sio.savemat('F:/PycharmProjects/J_spec/net_output/unet_output.mat', {'output_train': output,'output_test': output2})

def Test():
    Test_dir = 'F:/PycharmProjects/J_spec/data/data_Test/test_Jfft_1.mat'
    data_test = sio.loadmat(Test_dir)
    Test_input = data_test.get('X_J_fft_com')
    length = Test_input.shape[1]
    Test_input = np.expand_dims(Test_input, 0)

    Test_label = data_test.get('X_ps')
    Test_label = np.expand_dims(Test_label, -1)
    Test_label = np.expand_dims(Test_label, 0)

    x = tf.placeholder(tf.float32,shape = [None,None,None,2])
    y_ = tf.placeholder(tf.float32,shape = [None,None,None,1])

    y,lay1,lay3,lay5,layup1 = inference_unet_noise.net(x,len=length)

    # loss = tf.reduce_mean(tf.square(y - y_))
    # loss = lib_loss.huber_loss(y, y_, delta=1.0)

    variables_to_restore = []
    for v in tf.global_variables():
        variables_to_restore.append(v)
    saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V1, max_to_keep=None)
    tf.global_variables_initializer().run()
    saver.restore(sess, "F:\PycharmProjects\J_spec\lib_savenet\save_unet_Jfft001_noise\conv_unet1049999.ckpt-done")
    output = sess.run(y, feed_dict={x:Test_input})
    layer1 = sess.run(lay1, feed_dict={x:Test_input})
    layer3 = sess.run(lay3, feed_dict={x:Test_input})
    layer5 = sess.run(lay5, feed_dict={x:Test_input})
    layerup1 = sess.run(layup1, feed_dict={x:Test_input})
    output = np.squeeze(output)
    # loss_train = sess.run(loss, feed_dict={x: Test_input, y_: Test_label})
    # print('loss_train: %g' % (loss_train))
    sio.savemat('F:/PycharmProjects/J_spec/net_output/unet_output_Test.mat', {'output_Test': output,'layer1':layer1,'layer3':layer3,'layer5':layer5,'layerup1':layerup1})


# train_test()
Test()