import numpy as np
import scipy.io
from scipy.signal import convolve2d
import scipy.misc
import scipy.io as sio
import model
import tensorflow as tf
import numpy
import math

import os
import argparse
import random
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
epochnum =500
batch_size=4
height=64
width=64
depth=16
model_path =('/data3/zhy/gan_3d/gan_model_nogp')

maginput = tf.placeholder(dtype=tf.float32, shape=[batch_size, depth,height, width, 1])
fieldinput = tf.placeholder(dtype=tf.float32, shape=[batch_size, depth,height, width, 1])
tkdinput = tf.placeholder(dtype=tf.float32, shape=[batch_size, depth,height, width, 1])
coslabel = tf.placeholder(dtype=tf.float32, shape=[batch_size, depth,height, width, 1])

cosout = model.synthesis_nogp(maginput,fieldinput,tkdinput)
dis_coslabel = model.discriminator_s(coslabel,reuse=False)
dis_cosout = model.discriminator_s(cosout,reuse=True)
dis_loss=model.dloss(dis_cosout,dis_coslabel)
gen_loss,mseloss,dloss,gdloss=model.gloss_2(cosout,coslabel,dis_cosout)

# summary_op1 = tf.summary.scalar('trainloss', loss)
# summary_op2 = tf.summary.scalar('testloss', loss)

variable_to_gen = []
for variable in tf.trainable_variables():
    if (variable.name.startswith('syn')):
        variable_to_gen.append(variable)
variable_to_dis = []
for variable in tf.trainable_variables():
    if (variable.name.startswith('dis')):
        variable_to_dis.append(variable)
saver1 = tf.train.Saver(variable_to_dis, write_version=tf.train.SaverDef.V1, max_to_keep=None)
gen_op = tf.train.AdamOptimizer(0.00001).minimize(gen_loss, var_list=variable_to_gen)
dis_op = tf.train.AdamOptimizer(0.00001).minimize(dis_loss, var_list=variable_to_dis)
variables_to_restore = []
for v in tf.global_variables():
    variables_to_restore.append(v)
saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V1, max_to_keep=None)
with tf.Session() as sess:
    # writer = tf.summary.FileWriter('./my_graph/train')
    # writer1 = tf.summary.FileWriter('./my_graph/test')
    sess.run(tf.initialize_all_variables())
    #saver1.restore(sess,"/data1/zhy/models/train1024/synthesis_bn_20.ckpt-done")
    last_file = tf.train.latest_checkpoint(model_path)
    if last_file:
        tf.logging.info('Restoring model from {}'.format(last_file))
        saver.restore(sess, last_file)
        # saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V1)
        #saver.restore(sess, "/data1/zhy/models/final/synthesis_f_2.ckpt-done")

    saver.restore(sess, "/data3/zhy/gan_3d/gan_model_nogp/syn6.ckpt-done")
    #saver1.restore(sess, "/data3/zhy/gan_3d/gan_modell2/syn_8.ckpt-done")
    for epoch in range(epochnum):
        sumnum = []
        for i in range(160, 16000):
            sumnum.append(i)
        sumnum.remove(13944)
        # sumnum.remove(9267)
        # sumnum.remove(12931)
        # sumnum.remove(6957)
        # sumnum.remove(1604)
        for x in range(1,3959):
            sample=random.sample(sumnum,4)
            sumnum.remove(sample[0])
            sumnum.remove(sample[1])
            sumnum.remove(sample[2])
            sumnum.remove(sample[3])

            i=sample[0]
            i2 = sample[1]
            i3 = sample[2]
            i4 = sample[3]
            #print(i,i2,i3,i4)
            a1 = epoch + 1


            mimg1 = model.load_magimage('/data3/zhy/data/mag_or/mag%d.mat' % i)
            mimg2 = model.load_magimage('/data3/zhy/data/mag_or/mag%d.mat' % i2)
            mimg3 = model.load_magimage('/data3/zhy/data/mag_or/mag%d.mat' % i3)
            mimg4 = model.load_magimage('/data3/zhy/data/mag_or/mag%d.mat' % i4)
            _maginput=np.concatenate([mimg1,mimg2,mimg3,mimg4],0)

            fimg1 =  model.load_fieldimage('/data3/zhy/data/field_or/field%d.mat' % i)
            fimg2 =  model.load_fieldimage('/data3/zhy/data/field_or/field%d.mat' % i2)
            fimg3 =  model.load_fieldimage('/data3/zhy/data/field_or/field%d.mat' % i3)
            fimg4 =  model.load_fieldimage('/data3/zhy/data/field_or/field%d.mat' % i4)
            _fieldinput = np.concatenate([fimg1, fimg2, fimg3, fimg4], 0)

            cimg1 = model.load_cosmos('/data3/zhy/data/cosmos_or/cos%d.mat' % i)
            cimg2 = model.load_cosmos('/data3/zhy/data/cosmos_or/cos%d.mat' % i2)
            cimg3 = model.load_cosmos('/data3/zhy/data/cosmos_or/cos%d.mat' % i3)
            cimg4 = model.load_cosmos('/data3/zhy/data/cosmos_or/cos%d.mat' % i4)
            _cosinput = np.concatenate([cimg1, cimg2, cimg3, cimg4], 0)

            tkd1 = model.load_tkd('/data3/zhy/data/tkd/tkd%d.mat' % i)
            tkd2 = model.load_tkd('/data3/zhy/data/tkd/tkd%d.mat' % i2)
            tkd3 = model.load_tkd('/data3/zhy/data/tkd/tkd%d.mat' % i3)
            tkd4 = model.load_tkd('/data3/zhy/data/tkd/tkd%d.mat' % i4)
            _tkdinput = np.concatenate([tkd1, tkd2, tkd3, tkd4], 0)
            for i in range(2):
                sess.run(dis_op,
                     feed_dict={maginput: _maginput, fieldinput: _fieldinput, tkdinput: _tkdinput, coslabel: _cosinput})
            sess.run(gen_op,
                     feed_dict={maginput: _maginput, fieldinput: _fieldinput, tkdinput: _tkdinput,coslabel:_cosinput})
            if x % 20 == 0:
                print(a1)
                mseloss_1 = sess.run(mseloss,
                                 feed_dict={maginput: _maginput, fieldinput: _fieldinput, tkdinput: _tkdinput,coslabel:_cosinput})

                dloss_1 = sess.run(dloss,
                                 feed_dict={maginput: _maginput, fieldinput: _fieldinput, tkdinput: _tkdinput,
                                            coslabel: _cosinput})
                gdloss_1 = sess.run(gdloss,
                                  feed_dict={maginput: _maginput, fieldinput: _fieldinput, tkdinput: _tkdinput,
                                             coslabel: _cosinput})
                genloss_1 = sess.run(gen_loss,
                                  feed_dict={maginput: _maginput, fieldinput: _fieldinput, tkdinput: _tkdinput,
                                             coslabel: _cosinput})
                loss_4 = sess.run(dis_loss,
                                  feed_dict={maginput: _maginput, fieldinput: _fieldinput, tkdinput: _tkdinput,
                                             coslabel: _cosinput})



                mimg1 = model.load_magimage('/data3/zhy/data/mag_or/mag25.mat')
                mimg2 = model.load_magimage('/data3/zhy/data/mag_or/mag25.mat')
                mimg3 = model.load_magimage('/data3/zhy/data/mag_or/mag25.mat')
                mimg4 = model.load_magimage('/data3/zhy/data/mag_or/mag25.mat')
                _maginput = np.concatenate([mimg1, mimg2, mimg3, mimg4], 0)

                fimge1 = model.load_fieldimage('/data3/zhy/data/field_or/field25.mat')
                fimge2 = model.load_fieldimage('/data3/zhy/data/field_or/field25.mat')
                fimge3= model.load_fieldimage('/data3/zhy/data/field_or/field25.mat')
                fimge4 = model.load_fieldimage('/data3/zhy/data/field_or/field25.mat')
                _fieldinput = np.concatenate([fimg1, fimg2, fimg3, fimg4], 0)

                tkd1 = model.load_tkd('/data3/zhy/data/tkd/tkd25.mat' )
                tkd2 = model.load_tkd('/data3/zhy/data/tkd/tkd25.mat')
                tkd3 = model.load_tkd('/data3/zhy/data/tkd/tkd25.mat')
                tkd4 = model.load_tkd('/data3/zhy/data/tkd/tkd25.mat')
                _tkdinput = np.concatenate([tkd1, tkd2, tkd3, tkd4], 0)

                cimg1= model.load_cosmos('/data3/zhy/data/cosmos_or/cos25.mat')
                cimg2 = model.load_cosmos('/data3/zhy/data/cosmos_or/cos25.mat')
                cimg3 = model.load_cosmos('/data3/zhy/data/cosmos_or/cos25.mat')
                cimg4 = model.load_cosmos('/data3/zhy/data/cosmos_or/cos25.mat')


                _cosinput = np.concatenate([cimg1, cimg2, cimg3, cimg4], 0)




                timag = sess.run(cosout,
                                 feed_dict={maginput: _maginput, fieldinput: _fieldinput, tkdinput: _tkdinput })
                timag1 = timag[0, :, :, :,:, ]
                tmagimg11 = timag1[:, :,:, 0]
                tmagimg12 = tmagimg11[15, :, :, ]
                tmagimg = model.pre_cos_ssim(tmagimg12)

                timag2 = timag[1, :, :, :, :,]
                tmagimg22 = timag2[:, :,:, 0]
                tmagimg23 = tmagimg22[11, :, :,]
                tmagimg2 = model.pre_cos_ssim(tmagimg23)

                timag3 = timag[2, :, :, :,:, ]
                tmagimg33 = timag3[:, :,:, 0]
                tmagimg34 = tmagimg33[8, :, :, ]
                tmagimg3 = model.pre_cos_ssim(tmagimg34)

                timag4 = timag[3, :, :, :, :,]
                tmagimg44 = timag4[:, :, :,0]
                tmagimg45 = tmagimg44[3, :, :, ]
                tmagimg4 = model.pre_cos_ssim(tmagimg45)


                mag21 = model.pre_cos_ssim(model.quweidu(cimg1)[15, :, :,])
                mag41 = model.pre_cos_ssim(model.quweidu(cimg2)[11, :, :,])
                mag61 = model.pre_cos_ssim(model.quweidu(cimg3)[8, :, :,])
                mag81 = model.pre_cos_ssim(model.quweidu(cimg4)[3, :, :,])

                ssim1 = model.compute_ssim(tmagimg, mag21)
                ssim2 = model.compute_ssim(tmagimg2, mag41)
                ssim3 = model.compute_ssim(tmagimg3, mag61)
                ssim4 = model.compute_ssim(tmagimg4, mag81)

                cosssim = (ssim1 + ssim2 + ssim3 + ssim4) / 4

                print('costepoch%d:' % x, genloss_1, mseloss_1, dloss_1,gdloss_1, loss_4)
                print(cosssim)
        if  epoch %1 == 0:
            #if cosssim>0.915:
            saver.save(sess, os.path.join(model_path, 'syn%d.ckpt-done' % epoch))
            tf.logging.info('Done training%d -- epoch limit reached' % epoch)




