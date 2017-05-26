import tensorflow as tf


weights = {'conv1': tf.Variable(tf.truncated_normal([7, 7, 1, 96])),
           'conv10': tf.Variable(tf.truncated_normal([1, 1, 1, 1000]))}

biases = {'conv1': tf.Variable(tf.truncated_normal([96])),
          'conv10': tf.Variable(tf.truncated_normal([1000]))}


def fire_module(input, fire_id, s1=16, e1=64, e3=64):
    fire_weights = {'conv_s_1': tf.Variable(tf.truncated_normal([1, 1, 1, s1])),
                    'conv_e_1': tf.Variable(tf.truncated_normal([1, 1, 1, e1])),
                    'conv_e_3': tf.Variable(tf.truncated_normal([3, 3, 1, e3]))}

    fire_biases = {'conv_s_1': tf.Variable(tf.truncated_normal([s1])),
                   'conv_e_1': tf.Variable(tf.truncated_normal([e1])),
                   'conv_e_3': tf.Variable(tf.truncated_normal([e3]))}

    with tf.variable_scope(fire_id):
        output = tf.nn.conv2d(input, fire_weights['conv'], strides=[1, 1, 1, 1], padding='SAME', name='conv_s_1')
        output = tf.nn.relu(tf.nn.bias_add(output, fire_biases['conv_s_1']))

        expand1 = tf.nn.conv2d(output, fire_weights['conv_e_1'], strides=[1, 1, 1, 1], padding='SAME', name='conv_e_1')
        expand1 = tf.nn.bias_add(expand1, fire_biases['conv_e_1'])

        expand3 = tf.nn.conv2d(output, fire_weights['conv_e_3'], strides=[1, 1, 1, 1], padding='SAME', name='conv_e_3')
        expand3 = tf.nn.bias_add(expand3, fire_biases['conv_e_3'])

        result = tf.concat([expand1, expand3], 3, name='concat_e1_e3')
        return tf.nn.relu(result)


def squeeze_net(input):
    output = tf.nn.conv2d(input, weights['conv1'], strides=[1,2,2,1], padding='SAME', name='conv1')
    output = tf.nn.bias_add(output, biases['conv1'])

    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool1')
    
    output = fire_module(output, s1=16, e1=64, e3=64, fire_id='fire2')
    output = fire_module(output, s1=16, e1=64, e3=64, fire_id='fire3')
    output = fire_module(output, s1=32, e1=128, e3=128, fire_id='fire4')

    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool4')

    output = fire_module(output, s1=32, e1=128, e3=128, fire_id='fire5')
    output = fire_module(output, s1=48, e1=192, e3=192, fire_id='fire6')
    output = fire_module(output, s1=48, e1=192, e3=192, fire_id='fire7')
    output = fire_module(output, s1=64, e1=256, e3=256, fire_id='fire8')

    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool8')

    output = fire_module(output, s1=64, e1=256, e3=256, fire_id='fire9')

    output = tf.nn.dropout(output, keep_prob=0.5, name='dropout9')

    output = tf.nn.conv2d(output, weights['conv10'], strides=[1, 1, 1, 1], padding='SAME', name='conv10')
    output = tf.nn.bias_add(output, biases['conv10'])

    output = tf.nn.avg_pool(output, ksize=[1, 13, 13, 1], strides=[1, 1, 1, 1], padding='SAME', name='avgpool10')

    return output