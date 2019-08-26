import numpy as np
import tensorflow as tf
import os
import network
import yuv_import
import argparse

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name")
parser.add_argument("--frames", type=int)
parser.add_argument("--height", type=int)
parser.add_argument("--width", type=int)

args = parser.parse_args()

Height = args.height
Width = args.width
frames = args.frames
name = args.name

Channel = 1
batch_size = 1
relu = 1
kernel = [5, 5]
filter_num = 24
CNNlayer = 5

filename1 = './' + name + '_raw.yuv' #raw
filename2 = './' + name + '_HEVC_QP42.yuv' #comp

step = 22

def generate_weight(x_out):

    u_weight = tf.expand_dims(x_out, axis=-1)
    u_weight = tf.expand_dims(u_weight, axis=-1)
    u_weight = tf.expand_dims(u_weight, axis=-1)

    u_weight = tf.tile(u_weight, [1, 1, Height, Width, 1])

    f_weight = tf.ones([batch_size, step, Height, Width, Channel]) - u_weight

    return f_weight, u_weight


def dense(x, step):

    for i in range(step):

        if i == 0:
            reuse = False
        else:
            reuse = True

        x_1 = x[:,i,:]

        x_2 = tf.layers.dense(x_1, 1, kernel_initializer = tf.random_normal_initializer, reuse = reuse, name = 'PQF')

        if i == 0:
            x_o = tf.expand_dims(x_2, 1)
        else:
            x_o = tf.concat([x_o, tf.expand_dims(x_2, 1)], axis=1)

    return x_o


def sig(x):

    a = tf.get_variable('W/a', initializer=10.0)

    y = 1/(1 + tf.exp(-x/a))

    return y, a


config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config = config)

x1 = tf.placeholder(tf.float32, [batch_size, step, Height, Width, Channel]) #raw
x2 = tf.placeholder(tf.float32, [batch_size, step, Height, Width, Channel]) #compressed

x_1 = tf.placeholder(tf.float32, [batch_size, step, 190]) #feat

cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=256)

x3, state = tf.nn.bidirectional_dynamic_rnn(cell, cell, x_1, dtype=x1.dtype, scope = 'PQF')

x33 = tf.concat([x3[0], x3[1]], axis=-1)

x_out, aa = sig(tf.squeeze(dense(x33, step), [-1]))

forget, update = generate_weight(x_out)

outputs = network.net_bi_wcell(x2, forget, update, step, Height, Width, filter_num, kernel, relu, CNNlayer, peephole = False, scale = False)

saver = tf.train.Saver()
saver.restore(sess, './QP42_model/model.ckpt-200000')

# num_params = 0
# for variable in tf.trainable_variables():
#     shape = variable.get_shape()
#     num_params += reduce(mul, [dim.value for dim in shape], 1)
#
# print(num_params)

feat = np.load(name + '_HEVC_QP42_quality_features.npy')

feat = np.concatenate((feat, feat[-1:]), axis=0)
feat = np.concatenate((feat, feat[-1:]), axis=0)

feature = np.zeros([batch_size, frames, 190])

for f in range(5):

    frame_begin = 0
    index = 0

    for s in range(frames):

        if frame_begin + s >= 2:

            for fe in range(38):

                m1 = min(feat[frame_begin + s - 2:frame_begin + s + 3, fe])
                m2 = max(feat[frame_begin + s - 2:frame_begin + s + 3, fe])

                if m2 > m1:
                    feat_norm = (feat[frame_begin + s - 2:frame_begin + s + 3, fe] - m1) / (m2 - m1)
                else:
                    feat_norm = [0.0, 0.0, 0.0, 0.0, 0.0]

                feature[index, s, 5 * fe: 5 * fe + 5] = feat_norm

        elif frame_begin + s == 1:

            for fe in range(38):

                m1 = min(feat[frame_begin + s - 1:frame_begin + s + 3, fe])
                m2 = max(feat[frame_begin + s - 1:frame_begin + s + 3, fe])

                if m2 > m1:
                    feat_norm = (feat[frame_begin + s - 1:frame_begin + s + 3, fe] - m1) / (m2 - m1)
                    feat_norm = np.concatenate((feat_norm[0:1], feat_norm), axis=0)
                else:
                    feat_norm = [0.0, 0.0, 0.0, 0.0, 0.0]

                feature[index, s, 5 * fe: 5 * fe + 5] = feat_norm

        else:

            for fe in range(38):

                m1 = min(feat[frame_begin + s:frame_begin + s + 3, fe])
                m2 = max(feat[frame_begin + s:frame_begin + s + 3, fe])

                if m2 > m1:
                    feat_norm = (feat[frame_begin + s :frame_begin + s + 3, fe] - m1) / (m2 - m1)
                    feat_norm = np.concatenate((feat_norm[0:1], feat_norm), axis=0)
                    feat_norm = np.concatenate((feat_norm[0:1], feat_norm), axis=0)
                else:
                    feat_norm = [0.0, 0.0, 0.0, 0.0, 0.0]

                feature[index, s, 5 * fe: 5 * fe + 5] = feat_norm

d_psnr = np.zeros([frames])

for seq in range(frames//(step-2)):

    if seq == 0:

        [Y_rawv, U0, V0] = yuv_import.yuv_import(filename1, (Height, Width),
                                                 step-1, 0)
        [Y_comp, U1, V1] = yuv_import.yuv_import(filename2, (Height, Width),
                                                 step-1, 0)

        Y_rawv = np.concatenate((Y_rawv[0:1], Y_rawv), axis=0)
        Y_comp = np.concatenate((Y_comp[0:1], Y_comp), axis=0)
        feature = np.concatenate((feature[:, 0:1, :], feature), axis=1)

    elif seq < frames // (step - 2) - 1:

        [Y_rawv, U0, V0] = yuv_import.yuv_import(filename1, (Height, Width),
                                                 step, seq*(step-2)-1)
        [Y_comp, U1, V1] = yuv_import.yuv_import(filename2, (Height, Width),
                                                 step, seq*(step-2)-1)

    elif seq == frames//(step-2) - 1:

        [Y_rawv, U0, V0] = yuv_import.yuv_import(filename1, (Height, Width),
                                                 step-1, seq * (step - 2)-1)
        [Y_comp, U1, V1] = yuv_import.yuv_import(filename2, (Height, Width),
                                                 step-1, seq * (step - 2)-1)
        Y_rawv = np.concatenate((Y_rawv, Y_rawv[-1:]), axis=0)
        Y_comp = np.concatenate((Y_comp, Y_comp[-1:]), axis=0)
        feature = np.concatenate((feature, feature[:, -1:, :]), axis=1)

    Y_comp = Y_comp[np.newaxis, :]
    Y_comp = Y_comp[:, :, :, :, np.newaxis] / 255.0

    Y_rawv = Y_rawv[np.newaxis, :]
    Y_rawv = Y_rawv[:, :, :, :, np.newaxis] / 255.0

    Y_enhanced = sess.run(outputs + x2, feed_dict={x2: Y_comp, x_1: feature[:, seq*(step-2):seq*(step-2) + step]})

    for f in range(1, step-1):

        mse = np.mean(np.power(np.subtract(Y_rawv[0, f:f + 1], Y_enhanced[0, f:f + 1]),2.0))
        mse0 = np.mean(np.power(np.subtract(Y_rawv[0, f:f + 1], Y_comp[0, f:f + 1]), 2.0))

        psnr = 10.0*np.log(1.0/mse)/np.log(10.0)
        psnr0 = 10.0 * np.log(1.0 / mse0) / np.log(10.0)

        d_psnr[seq*(step-2) + f - 1] = psnr - psnr0

        print('Frame: ' + str(seq*(step-2) + f) + '  Delta PSNR (dB): ' + str(psnr - psnr0))

print('Average Delta PSNR (dB):')
print(np.mean(d_psnr))


