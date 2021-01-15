import tensorflow as tf
from cell import QGConvLSTMCell

def CNN(x, step, filter_no, filter_no_last, kernel, relu, layer, scale, name):

    for i in range(step):

        if i == 0:
            reuse = False
        else:
            reuse = True

        x_i = x[:,i,:,:,:]

        if scale == True:

            x_i_1 = tf.layers.conv2d(x_i, filter_no // 3, [3, 3], padding='SAME', reuse=reuse,name='conv' + name + '_' + '01')
            x_i_2 = tf.layers.conv2d(x_i, filter_no // 3, [5, 5], padding='SAME', reuse=reuse,name='conv' + name + '_' + '02')
            x_i_3 = tf.layers.conv2d(x_i, filter_no // 3, [7, 7], padding='SAME', reuse=reuse,name='conv' + name + '_' + '03')
            x_i = tf.concat([x_i_1, x_i_2, x_i_3], axis=-1, name='conv' + name + '_' + str(0))

        elif scale == False:

            x_i = tf.layers.conv2d(x_i, filter_no, kernel, padding='SAME', reuse=reuse, name='conv' + name + '_' + str(0))

        if relu == 1:
            x_i = tf.nn.relu(x_i)

        for ii in range(1, layer - 1):

            x_i = tf.layers.conv2d(x_i, filter_no, kernel, padding='SAME', reuse=reuse, name='conv' + name + '_' + str(ii))

            if relu == 1:
                x_i = tf.nn.relu(x_i)

        x_i = tf.layers.conv2d(x_i, filter_no_last, kernel, padding='SAME', reuse=reuse, name='conv' + name + '_' + str(layer - 1))

        if relu == 1:
            if filter_no_last == filter_no:
                x_i = tf.nn.relu(x_i)

        if i == 0:
            x_o = tf.expand_dims(x_i, 1)
        else:
            x_o = tf.concat([x_o, tf.expand_dims(x_i, 1)], axis=1)

    return x_o

def net_bi_wcell(x, f, u, step, Height, Width, filter_num, kernel, relu, CNNlayer, peephole, scale):

    x1 = CNN(x, step, filter_num, filter_num, kernel, relu, CNNlayer, scale=scale, name="1")

    # print("CNN")

    inputs = tf.concat([x1, f, u], axis = -1)

    cell = QGConvLSTMCell(shape=[Height, Width], filters = filter_num, kernel = kernel, peephole = peephole)

    x2, state = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs, dtype = inputs.dtype)

    # print("LSTM")

    x22 = tf.concat([x2[0], x2[1]], axis=4)

    x3 = CNN(x22, step, filter_num, 1, kernel, relu, CNNlayer, scale=False, name="2")

    # print("CNN")

    return x3
#
# def net_bi_origcell(x, step, Height, Width, filter_num, kernel, relu, CNNlayer, peephole, scale):
#
#     x1 = CNN(x, step, filter_num, filter_num, kernel, relu, CNNlayer, scale=scale, name="1")
#
#     print("CNN")
#
#     cell = ConvLSTMCell_orig(shape=[Height, Width], filters=filter_num, kernel=kernel, peephole = peephole)
#
#     x2, state = tf.nn.bidirectional_dynamic_rnn(cell, cell, x1, dtype = x1.dtype)
#
#     print("LSTM")
#
#     x22 = tf.concat([x2[0], x2[1]], axis=4)
#
#     x3 = CNN(x22, step, filter_num, 1, kernel, relu, CNNlayer, scale=False, name="2")
#
#     print("CNN")
#
#     return x3





