import numpy as np
import tensorflow as tf

## TensorFlow helper functions

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'

def _relu(x, leakness=0.0, name=None):
    if leakness > 0.0:
        name = 'lrelu' if name is None else name
        return tf.maximum(x, x*leakness, name='lrelu')
    else:
        name = 'relu' if name is None else name
        return tf.nn.relu(x, name='relu')


def _dropout(x, keep_prob=1.0, name=None):
    assert keep_prob >= 0.0 and keep_prob <= 1.0
    if keep_prob == 1.0:
        return x
    else:
        return tf.nn.dropout(x, keep_prob, name=name)


def _conv(x, filter_size, out_channel, strides, pad='SAME', input_q=None, output_q=None, name='conv'):
    if (input_q == None)^(output_q == None):
        raise ValueError('Input/Output splits are not correctly given.')

    in_shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        # Main operation: conv2d
        kernel = tf.get_variable('kernel', [filter_size, filter_size, in_shape[3], out_channel],
                        tf.float32, initializer=tf.random_normal_initializer(
                            stddev=np.sqrt(1.0/filter_size/filter_size/in_shape[3])))
        if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (kernel.name, str(kernel.get_shape().as_list())))
        conv = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], pad)

        # Split and split loss
        if (input_q is not None) and (output_q is not None):
            _add_split_loss(kernel, input_q, output_q)

    return conv


def _conv_with_init(x, filter_size, out_channel, strides, pad='SAME', init_k=None, name='conv'):
    in_shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        # Main operation: conv2d
        if init_k is not None:
            initializer_k = tf.constant_initializer(init_k)
        else:
            initializer_k =tf.random_normal_initializer(stddev=np.sqrt(1.0/filter_size/filter_size/in_shape[3]))
        kernel = tf.get_variable('kernel', [filter_size, filter_size, in_shape[3], out_channel],
                        tf.float32, initializer=initializer_k)
        if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (kernel.name, str(kernel.get_shape().as_list())))
        conv = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], pad)

    return conv


def _fc(x, out_dim, input_q=None, output_q=None, name='fc'):
    if (input_q == None)^(output_q == None):
        raise ValueError('Input/Output splits are not correctly given.')

    with tf.variable_scope(name):
        # Main operation: fc
        w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
                        tf.float32, initializer=tf.random_normal_initializer(
                            stddev=np.sqrt(1.0/x.get_shape().as_list()[1])))
        if w not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, w)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (w.name, str(w.get_shape().as_list())))
        b = tf.get_variable('biases', [out_dim], tf.float32,
                            initializer=tf.constant_initializer(0.0))
        fc = tf.nn.bias_add(tf.matmul(x, w), b)

        # Split loss
        if (input_q is not None) and (output_q is not None):
            _add_split_loss(w, input_q, output_q)

    return fc


def _fc_with_init(x, out_dim, init_w=None, init_b=None, name='fc'):
    with tf.variable_scope(name):
        # Main operation: fc
        if init_w is not None:
            initializer_w = tf.constant_initializer(init_w)
        else:
            initializer_w = tf.random_normal_initializer(stddev=np.sqrt(1.0/x.get_shape().as_list()[1]))
        if init_b is not None:
            initializer_b = tf.constant_initializer(init_b)
        else:
            initializer_b = tf.constant_initializer(0.0)

        w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
                        tf.float32, initializer=initializer_w)
        b = tf.get_variable('biases', [out_dim], tf.float32,
                            initializer=initializer_b)
        if w not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, w)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (w.name, str(w.get_shape().as_list())))
        fc = tf.nn.bias_add(tf.matmul(x, w), b)

    return fc


def _get_split_q(ngroups, dim, name='split'):
    with tf.variable_scope(name):
        alpha = tf.get_variable('alpha', shape=[ngroups, dim], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(stddev=0.01))
        q = tf.nn.softmax(alpha, dim=0, name='q')

    return q

def _merge_split_q(q, merge_idxs, name='merge'):
    assert len(q.get_shape()) == 2
    ngroups, dim = q.get_shape().as_list()
    assert ngroups == len(merge_idxs)

    with tf.variable_scope(name):
        max_idx = np.max(merge_idxs)
        temp_list = []
        for i in range(max_idx + 1):
            temp = []
            for j in range(ngroups):
                if merge_idxs[j] == i:
                    temp.append(tf.slice(q, [j, 0], [1, dim]))
            temp_list.append(tf.add_n(temp))
        ret = tf.concat(temp_list, 0)

    return ret


def _get_even_merge_idxs(N, split):
    assert N >= split
    num_elems = [(N + split - i - 1)/split for i in range(split)]
    expand_split = [[i] * n for i, n in enumerate(num_elems)]
    return [t for l in expand_split for t in l]


def _add_split_loss(w, input_q, output_q):
    # Check input tensors' measurements
    assert len(w.get_shape()) == 2 or len(w.get_shape()) == 4
    in_dim, out_dim = w.get_shape().as_list()[-2:]
    assert len(input_q.get_shape()) == 2
    assert len(output_q.get_shape()) == 2
    assert in_dim == input_q.get_shape().as_list()[1]
    assert out_dim == output_q.get_shape().as_list()[1]
    assert input_q.get_shape().as_list()[0] == output_q.get_shape().as_list()[0]  # ngroups
    ngroups = input_q.get_shape().as_list()[0]
    assert ngroups > 1

    # Add split losses to collections
    T_list = []
    U_list = []
    if input_q not in tf.get_collection('OVERLAP_LOSS_WEIGHTS') \
            and not "concat" in input_q.op.name:
        tf.add_to_collection('OVERLAP_LOSS_WEIGHTS', input_q)
        print('\t\tAdd overlap & split loss for %s' % input_q.name)
        T_temp, U_temp = ([], [])
        for i in range(ngroups):
            for j in range(ngroups):
                if i <= j:
                    continue
                T_temp.append(tf.reduce_sum(input_q[i,:] * input_q[j,:]))
            U_temp.append(tf.square(tf.reduce_sum(input_q[i,:])))
        T_list.append(tf.reduce_sum(T_temp)/(float(in_dim*(ngroups-1))/float(2*ngroups)))
        U_list.append(tf.reduce_sum(U_temp)/(float(in_dim*in_dim)/float(ngroups)))
    if output_q not in tf.get_collection('OVERLAP_LOSS_WEIGHTS') \
            and not "concat" in output_q.op.name:
        print('\t\tAdd overlap & split loss for %s' % output_q.name)
        tf.add_to_collection('OVERLAP_LOSS_WEIGHTS', output_q)
        T_temp, U_temp = ([], [])
        for i in range(ngroups):
            for j in range(ngroups):
                if i <= j:
                    continue
                T_temp.append(tf.reduce_sum(output_q[i,:] * output_q[j,:]))
            U_temp.append(tf.square(tf.reduce_sum(output_q[i,:])))
        T_list.append(tf.reduce_sum(T_temp)/(float(out_dim*(ngroups-1))/float(2*ngroups)))
        U_list.append(tf.reduce_sum(U_temp)/(float(out_dim*out_dim)/float(ngroups)))
    if T_list:
        tf.add_to_collection('OVERLAP_LOSS', tf.add_n(T_list)/len(T_list))
    if U_list:
        tf.add_to_collection('UNIFORM_LOSS', tf.add_n(U_list)/len(U_list))

    S_list = []
    if w not in tf.get_collection('WEIGHT_SPLIT_WEIGHTS'):
        tf.add_to_collection('WEIGHT_SPLIT_WEIGHTS', w)

        ones_col = tf.ones((in_dim,), dtype=tf.float32)
        ones_row = tf.ones((out_dim,), dtype=tf.float32)
        if len(w.get_shape()) == 4:
            w_reduce = tf.reduce_mean(tf.square(w), [0, 1])
            w_norm = w_reduce
            std_dev = np.sqrt(1.0/float(w.get_shape().as_list()[0])**2/in_dim)
            # w_norm = w_reduce / tf.reduce_sum(w_reduce)
        else:
            w_norm = w
            std_dev = np.sqrt(1.0/float(in_dim))
            # w_norm = w / tf.sqrt(tf.reduce_sum(tf.square(w)))

        for i in range(ngroups):
            if len(w.get_shape()) == 4:
                wg_row = tf.transpose(tf.transpose(w_norm * tf.square(output_q[i,:])) * tf.square(ones_col - input_q[i,:]))
                wg_row_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_row, 1))) / (in_dim*np.sqrt(out_dim))
                wg_col = tf.transpose(tf.transpose(w_norm * tf.square(ones_row - output_q[i,:])) * tf.square(input_q[i,:]))
                wg_col_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_col, 0))) / (np.sqrt(in_dim)*out_dim)
            else:  # len(w.get_shape()) == 2
                wg_row = tf.transpose(tf.transpose(w_norm * output_q[i,:]) * (ones_col - input_q[i,:]))
                wg_row_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_row * wg_row, 1))) / (in_dim*np.sqrt(out_dim))
                wg_col = tf.transpose(tf.transpose(w_norm * (ones_row - output_q[i,:])) * input_q[i,:])
                wg_col_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_col * wg_col, 0))) / (np.sqrt(in_dim)*out_dim)
            S_list.append(wg_row_l2 + wg_col_l2)
        # S = tf.add_n(S_list)/((ngroups-1)/ngroups)
        S = tf.add_n(S_list)/(2*(ngroups-1)*std_dev/ngroups)
        tf.add_to_collection('WEIGHT_SPLIT', S)

        # Add histogram for w if split losses are added
        scope_name = tf.get_variable_scope().name
        tf.summary.histogram("%s/" % scope_name, w)
        print('\t\tAdd split loss for %s(%dx%d, %d groups)' \
            % (tf.get_variable_scope().name, in_dim, out_dim, ngroups))

    return


def _bn(x, is_train, global_step=None, name='bn', no_scale=False):
    moving_average_decay = 0.9
    # moving_average_decay = 0.99
    # moving_average_decay_init = 0.99
    with tf.variable_scope(name):
        decay = moving_average_decay
        # if global_step is None:
            # decay = moving_average_decay
        # else:
            # decay = tf.cond(tf.greater(global_step, 100)
                            # , lambda: tf.constant(moving_average_decay, tf.float32)
                            # , lambda: tf.constant(moving_average_decay_init, tf.float32))
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
                        initializer=tf.zeros_initializer(), trainable=False)
        sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
                        initializer=tf.ones_initializer(), trainable=False)
        beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
                        initializer=tf.zeros_initializer())
        gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
                        initializer=tf.ones_initializer(), trainable=(not no_scale))
        # BN when training
        update = 1.0 - decay
        # with tf.control_dependencies([tf.Print(decay, [decay])]):
            # update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_sigma = sigma.assign_sub(update*(sigma - batch_var))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)

        mean, var = tf.cond(is_train, lambda: (batch_mean, batch_var),
                            lambda: (mu, sigma))
        bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)

        # bn = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-5)

        # bn = tf.contrib.layers.batch_norm(inputs=x, decay=decay,
                                          # updates_collections=[tf.GraphKeys.UPDATE_OPS], center=True,
                                          # scale=True, epsilon=1e-5, is_training=is_train,
                                          # trainable=True)
    return bn


## Other helper functions



