import tensorflow as tf
import dataset
import networks.model as model
import numpy as np
import os
import argparse
import cv2 as cv
import scipy.misc
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True  # 程序按需申请内存
sess = tf.InteractiveSession(config=config)

parser = argparse.ArgumentParser()
parser.add_argument("--train_file", default="./data_face/train_HR")
parser.add_argument("--test_file", default="./data_face/test_HR")
parser.add_argument("--nubBlocks", default=16, type=int)
parser.add_argument("--SPFILTER_DIM", default=64, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--savenet_path", default='./libSaveNet/savenet/')
parser.add_argument("--epoch", default=200000, type=int)
parser.add_argument("--learning_rate", default=0.0001, type=float)
parser.add_argument("--num_train", default=10000, type=int)
parser.add_argument("--num_test", default=1500, type=int)
parser.add_argument("--EPS", default=1e-12, type=float)
parser.add_argument("--train_path", default='./data/train_fft.mat')
parser.add_argument("--test_path", default='./data/test_fft.mat')
parser.add_argument("--fea_name", default='train_fea')
parser.add_argument("--label_name", default='train_lable')

args = parser.parse_args()

def train(args):
    train_xinp,train_yrl = dataset.data_load(args.train_path,'train_fea','train_lable')
    test_xinp,test_yrl = dataset.data_load(args.train_path,'train_fea','train_lable')
    train_xinp = np.expand_dims(train_xinp,1)
    train_yrl = np.expand_dims(train_yrl,-1)
    train_yrl = np.expand_dims(train_yrl,1)
    test_xinp = np.expand_dims(test_xinp,1)
    test_yrl = np.expand_dims(test_yrl,1)
    test_yrl = np.expand_dims(test_yrl,-1)
    x = tf.placeholder(tf.float32,shape = [args.batch_size,1,256, 2])
    y_ = tf.placeholder(tf.float32,shape = [args.batch_size,1,256,1])

    y = model.inference(x,args=args)
    loss = tf.reduce_mean(tf.abs(y - y_))


    tf.summary.scalar('loss', loss)
    summary_op = tf.summary.merge_all()

    batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    with tf.control_dependencies([batch_norm_updates_op]):
        train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
    variables_to_restore = []
    for v in tf.global_variables():
        variables_to_restore.append(v)
    saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2, max_to_keep=20)

    train_writer = tf.summary.FileWriter('./my_graph/train', sess.graph)
    test_writer = tf.summary.FileWriter('./my_graph/test')
    tf.global_variables_initializer().run()
    # last_file = tf.train.latest_checkpoint(savenet_path)
    # if last_file:
    #     tf.logging.info('Restoring model from {}'.format(last_file))
        # saver.restore(sess, last_file)
    count, m = 0, 0
    data_size = np.shape(train_xinp)
    for ep in range(args.epoch):
        batch_idxs = data_size[0] // args.batch_size
        for idx in range(batch_idxs):
            batch_input = train_xinp[idx * args.batch_size: (idx + 1) * args.batch_size]
            batch_labels = train_yrl[idx * args.batch_size: (idx + 1) * args.batch_size]
            # batch_input, batch_labels = dataset.random_batch(x_train,y_train,args.batch_size)

            sess.run(train_step, feed_dict={x: batch_input, y_: batch_labels})
            count += 1
            # print(count)
            if count % 100 == 0:
                m += 1
                # batch_input_test, batch_labels_test = dataset.random_batch(x_test, y_test, args.batch_size)
                batch_input_test = test_xinp[0 : args.batch_size]
                batch_labels_test = test_yrl[0 : args.batch_size]
                loss1 = sess.run(loss, feed_dict={x: batch_input,y_: batch_labels})
                loss2 = sess.run(loss, feed_dict={x: batch_input_test, y_: batch_labels_test})
                print("Epoch: [%2d], step: [%2d], train_loss: [%.8f]" \
                      % ((ep + 1), count, loss1), "\t", 'test_loss:[%.8f]' % (loss2))
                train_writer.add_summary(sess.run(summary_op, feed_dict={x: batch_input, y_: batch_labels}), m)
                test_writer.add_summary(sess.run(summary_op, feed_dict={x: batch_input_test,
                                                                     y_: batch_labels_test}), m)
            if (count + 1) % 10000 == 0:
                saver.save(sess, os.path.join(args.savenet_path, 'conv_ResNet%d.ckpt-done' % (count)))
def test(args):
    w1 = np.linspace(0, 2 * np.pi, 256, endpoint=True)
    savepath = './libSaveNet/savenet/'
    test_xinp,test_yrl = dataset.data_load(args.train_path,'train_fea','train_lable')
    test_xinp = np.expand_dims(test_xinp,1)
    test_yrl = np.expand_dims(test_yrl,1)
    test_yrl = np.expand_dims(test_yrl,-1)
    x = tf.placeholder(tf.float32,shape = [args.batch_size,1,256, 2])
    y_ = tf.placeholder(tf.float32,shape = [args.batch_size,1,256,1])
    y = model.ResNet(x,args=args)
    loss = tf.reduce_mean(tf.abs(y - y_))
    variables_to_restore = []
    for v in tf.global_variables():
        variables_to_restore.append(v)
    saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2, max_to_keep=20)
    tf.global_variables_initializer().run()
    saver.restore(sess, savepath)

    batch_input = test_xinp[0]
    batch_labels = test_yrl[0]
    output = sess.run(y, feed_dict={x: batch_input, y_: batch_labels})
    loss_test = sess.run(loss, feed_dict={x: batch_input, y_: batch_labels})
    output = np.squeeze(output)
    plt.figure()
    plt.title('test')
    plt.plot(w1, output)
    plt.show()

if __name__ == '__main__':
    train(args)