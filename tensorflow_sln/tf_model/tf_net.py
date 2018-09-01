import tensorflow as tf
import numpy as np
import random


# config tensorflow start session
def start_sess():
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


class Cnn(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.batch_size = 32
        self.epochs = 150
        self.learning_rate = 0.01

    # add convolution layer, the pooling layer, and the batch_norm layer after(is training for this)
    @staticmethod
    def conv_pool_layer(pre_layer, filters, conv_ksize, conv_strides, po_ksize, po_strides,
                        activation, padding='VALID', with_bn=False):
        conv_layer = tf.layers.conv2d(pre_layer, filters, conv_ksize, conv_strides, padding, activation=activation)
        pool_layer = tf.layers.max_pooling2d(conv_layer, po_ksize, po_strides)
        layer = tf.layers.batch_normalization(pool_layer, training=with_bn)
        return layer

    # add fully-connect layer(dense layer)
    @staticmethod
    def fully_con_layer(pre_layer, num_units, activation, with_bn=False):
        layer = tf.layers.dense(pre_layer, num_units, use_bias=True, activation=None)
        layer = tf.layers.batch_normalization(layer, training=with_bn)
        if activation is None:
            return layer
        else:
            return activation(layer)

    # calculate the product of a multi-dimension input, just like flattening a image
    # for placeholder: (None, 326*490*3) --> (326*490*3, )
    @staticmethod
    def dim_prod(dim_arr):
        d_arr = [i for i in dim_arr if i is not None or i == 1]
        return np.prod(d_arr)

    # split train data to mini batches
    @staticmethod
    def batchify(x, y, size):
        samples = x.shape[0]
        batches = []
        for i in range(0, samples, size):
            batches.append((x[i:i + size], y[i:i + size]))
        random.shuffle(batches)
        return batches

    def build_net(self):
        im_rows, im_cols, channels = self.input_shape[0], self.input_shape[1], self.input_shape[2]
        x_hold = tf.placeholder(tf.float32, shape=[None, im_rows*im_cols*channels], name='x_hold')
        y_hold = tf.placeholder(tf.float32, shape=[None, 2], name='y_hold')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        with_bn = tf.placeholder(tf.bool, name='with_bn')

        x_im = tf.reshape(x_hold, [-1, im_rows, im_cols, channels])
        relu = tf.nn.relu

        layer1 = self.conv_pool_layer(pre_layer=x_im, filters=8, conv_ksize=3, conv_strides=(2, 2), po_ksize=4,
                                        po_strides=4, activation=relu, padding='VALID')
        layer2 = self.conv_pool_layer(pre_layer=layer1, filters=16, conv_ksize=4, conv_strides=(1, 1), po_ksize=3,
                                        po_strides=3, activation=relu, padding='VALID')
        layer3 = self.conv_pool_layer(pre_layer=layer2, filters=32, conv_ksize=4, conv_strides=(1, 1), po_ksize=2,
                                        po_strides=2, activation=relu, padding='VALID', with_bn=with_bn)
        layer3_flat = tf.layers.flatten(layer3)

        layer4 = self.fully_con_layer(layer3_flat, 512, activation=relu, with_bn=with_bn)
        layer4_drop = tf.layers.dropout(layer4)

        layer5 = self.fully_con_layer(layer4_drop, 256, activation=relu, with_bn=with_bn)
        layer5_drop = tf.layers.dropout(layer5)

        layer6 = self.fully_con_layer(layer5_drop, 128, activation=relu, with_bn=with_bn)

        layer7 = self.fully_con_layer(layer6, 32, activation=relu, with_bn=with_bn)

        pred = self.fully_con_layer(layer7, 2, activation=None)

        return pred, x_hold, y_hold, keep_prob, with_bn

    def train_test(self, pred, x_hold, y_hold, keep_prob, with_bn, splited_data, save_path=None):
        x_train, x_test = splited_data[0], splited_data[1]
        y_train, y_test = splited_data[2], splited_data[3]
        sess = start_sess()

        # loss function definition
        error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(pred - y_hold), axis=1)), axis=0)

        # important step, update the distribution of mean & conv value when training
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(error)

        # calculate the accuracy of giving train/test data
        correct_pre = tf.divide(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(pred, y_hold)), axis=1)), 255) < 0.05
        accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

        # flatten the input image for the placeholder
        flat_shape = self.dim_prod(x_hold._shape_as_list())
        x_train = x_train.reshape(x_train.shape[0], flat_shape)
        x_test = x_test.reshape(x_test.shape[0], flat_shape)

        # mini_batch gradient descent
        batches = self.batchify(x_train, y_train, self.batch_size)

        # save model for reload
        saver =  tf.train.Saver(tf.global_variables())
        max_train_acc, max_test_acc, max_acc_epoch = 0.0, 0.0, 0  # to compare and choose the model with best accuracy
        tf.add_to_collection('predict', pred)  # pass 'pred' out for the test operation later

        print('Starting training session')
        sess.run(tf.global_variables_initializer())
        for i in range(self.epochs):
            for x_batch, y_batch in batches:
                 # add dropout and bn when training
                sess.run(train_step, feed_dict={x_hold: x_batch, y_hold: y_batch, keep_prob: 0.5, with_bn: True})
            # check training/test accuracy without dropout and bn
            train_acc = accuracy.eval(session=sess,feed_dict={x_hold: x_train, y_hold: y_train, keep_prob: 1., with_bn: False})
            test_acc = accuracy.eval(session=sess, feed_dict={x_hold: x_test, y_hold: y_test, keep_prob: 1, with_bn: False})
            print('epoch {0}'.format(i+1) + ':\t train acc:' + str(round(train_acc, 4)) + ';\t' + 'test acc:' + str(round(test_acc, 4)))

            # save the best model during training
            if max_train_acc*0.3+max_test_acc*0.7<train_acc * 0.3 + test_acc * 0.7:  # give a score for model in each epoch
                max_train_acc, max_test_acc, max_acc_epoch = train_acc, test_acc, i+1
                if not save_path is None:
                    saver.save(sess, save_path=save_path)
        print('Best model occurs in epoch {0}, train acc is {1:.4f}, test acc is {2:.4f}'.format(max_acc_epoch, max_train_acc, max_test_acc))
        sess.close()


# Tips: sess/.eval()等操作,占用计算资源，在代码中尽量少使用

