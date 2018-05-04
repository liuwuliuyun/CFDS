import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.framework as framework


weight_decay=0


def global_avg(x):
    with tf.variable_scope('global_avg'):
        net=tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
        return net


def relu(x, name='relu6'):
    return tf.nn.relu6(x, name)


def batch_norm(x, momentum=0.9, epsilon=1e-5, train=True, name='bn'):
    return tf.layers.batch_normalization(x,
        momentum=momentum,
        epsilon=epsilon,
        scale=True,
        training=train,
        name=name)


def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, wd=4e-5, name='conv2d', bias=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv


def conv2d_block(input, out_dim, k, s, is_train, name):
    with tf.variable_scope(name):
        net = conv2d(input, out_dim, k, k, s, s, name='conv2d')
        net = batch_norm(net, train=is_train, name='bn')
        net = relu(net)
        return net


def conv_1x1(input, output_dim, name, bias=False):
    return conv2d(input, output_dim, 1,1,1,1, stddev=0.02, name=name, bias=bias)


def pwise_block(input, output_dim, is_train, name, bias=False):
    with tf.variable_scope(name):
        out=conv_1x1(input, output_dim, bias=bias, name='pwb')
        out=batch_norm(out, train=is_train, name='bn')
        out=relu(out)
        return out


def dwise_conv(input, k_h=3, k_w=3, channel_multiplier= 1, strides=[1,1,1,1],
    padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
    with tf.variable_scope(name):
        in_channel=input.get_shape().as_list()[-1]
        w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None,name=None,data_format=None)
        if bias:
            biases = tf.get_variable('bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv


def res_block(input, expansion_ratio, output_dim, stride, is_train, name, bias=False, shortcut=True):
    with tf.variable_scope(name):
        bottleneck_dim=round(expansion_ratio*input.get_shape().as_list()[-1])
        net = conv_1x1(input, bottleneck_dim, name='pw', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_bn')
        net = relu(net)
        net = dwise_conv(net, strides=[1, stride, stride, 1], name='dw', bias=bias)
        net = batch_norm(net, train=is_train, name='dw_bn')
        net = relu(net)
        net = conv_1x1(net, output_dim, name='pw_linear', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_linear_bn')
        if shortcut and stride == 1:
            in_dim=int(input.get_shape().as_list()[-1])
            if in_dim != output_dim:
                ins=conv_1x1(input, output_dim, name='ex_dim')
                net=ins+net
            else:
                net=input+net
        return net


def network(x, name, embedding_size=256, width_mul=1, wd=0.):
    is_training = False
    global weight_decay
    weight_decay = wd
    with tf.name_scope(name):
        exp = 6  # expansion ratio
        with tf.variable_scope('mobilenet', reuse=tf.AUTO_REUSE):
            x = (x - 127.5) * 0.0078125
            x = tf.image.resize_images(x, (224, 224))
            cands = []
            net = conv2d_block(x, 32 * width_mul, 3, 2, is_training, name='conv1_1')  # size/2
            net = res_block(net, 1, 16 * width_mul, 1, is_training, name='res2_1')
            cands.append(net)
            net = res_block(net, exp, 24 * width_mul, 2, is_training, name='res3_1')  # size/4
            cands.append(net)
            net = res_block(net, exp, 24 * width_mul, 1, is_training, name='res3_2')
            cands.append(net)
            net = res_block(net, exp, 32 * width_mul, 2, is_training, name='res4_1')  # size/8
            cands.append(net)
            net = res_block(net, exp, 32 * width_mul, 1, is_training, name='res4_2')
            cands.append(net)
            net = res_block(net, exp, 32 * width_mul, 1, is_training, name='res4_3')
            cands.append(net)
            net = res_block(net, exp, 64 * width_mul, 1, is_training, name='res5_1')
            cands.append(net)
            net = res_block(net, exp, 64 * width_mul, 1, is_training, name='res5_2')
            cands.append(net)
            net = res_block(net, exp, 64 * width_mul, 1, is_training, name='res5_3')
            cands.append(net)
            net = res_block(net, exp, 64 * width_mul, 1, is_training, name='res5_4')
            cands.append(net)
            net = res_block(net, exp, 96 * width_mul, 2, is_training, name='res6_1')  # size/16
            cands.append(net)
            net = res_block(net, exp, 96 * width_mul, 1, is_training, name='res6_2')
            cands.append(net)
            net = res_block(net, exp, 96 * width_mul, 1, is_training, name='res6_3')
            cands.append(net)
            net = res_block(net, exp, 160 * width_mul, 2, is_training, name='res7_1')  # size/32
            cands.append(net)
            net = res_block(net, exp, 160 * width_mul, 1, is_training, name='res7_2')
            cands.append(net)
            net = res_block(net, exp, 160 * width_mul, 1, is_training, name='res7_3')
            cands.append(net)
            net = res_block(net, exp, 320 * width_mul, 1, is_training, name='res8_1', shortcut=False)
            cands.append(net)
            net = pwise_block(net, 1280 * width_mul, is_training, name='conv9_1')
            net = global_avg(net)
            net = layers.flatten(net)
            net = layers.fully_connected(
                net, embedding_size,
                activation_fn=None,
                weights_initializer=layers.variance_scaling_initializer(mode='FAN_OUT'),
                weights_regularizer=layers.l2_regularizer(wd))
            net = batch_norm(net, train=is_training, name='final_bn') 
            y = tf.nn.l2_normalize(net, 1)
            return y


class extractor:
    def __init__(self, session, devices, batch_size):
        self.session = session
        assert batch_size % len(devices) == 0
        batch_size_per_device = batch_size // len(devices)
        with tf.name_scope('extactor'):
            with tf.device('/cpu:0'):
                self.images = tf.placeholder(
                    tf.float32, (batch_size, 112, 112, 3), 'images')
                embedding = network(
                    tf.zeros((1, 112, 112, 3)), 'network_cpu')
                embeddings = []
            for i, device in enumerate(devices):
                with tf.device(device):
                    local_embeddings = network(
                        self.images[i * batch_size_per_device: (i + 1) * batch_size_per_device], 'network_device%d' % i)
                    embeddings.append(local_embeddings)
            with tf.device('/cpu:0'):
                self.embeddings = tf.concat(embeddings, axis=0)
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mobilenet'))
        saver.restore(session, '/SSH/demo_0/models/extractor')
    def extract(self, images):
        e = self.session.run(self.embeddings, feed_dict={self.images: images})
        return [i for i in e]

        
