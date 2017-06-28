from collections import namedtuple

import tensorflow as tf
import numpy as np

import utils


HParams = namedtuple('HParams',
                    'batch_size, num_classes, num_residual_units, k, weight_decay, momentum, finetune, '
                    'ngroups1, ngroups2, ngroups3, split_params')

class ResNet(object):
    def __init__(self, hp, images, labels, global_step, name=None, reuse_weights=False):
        self._hp = hp # Hyperparameters
        self._images = images # Input image
        self._labels = labels # Input labels
        self._global_step = global_step
        self.lr = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)
        self._counted_scope = []
        self._flops = 0
        self._weights = 0

    def build_model(self):
        print('Building model')
        filters = [16, 16 * self._hp.k, 32 * self._hp.k, 64 * self._hp.k]
        strides = [1, 2, 2]

        # Init. conv.
        print('\tBuilding unit: init_conv')
        x = utils._conv(self._images, 3, filters[0], 1, name='init_conv')

        # unit_1_x
        x = self._residual_block_first(x, filters[1], strides[0], name='unit_1_0')
        x = self._residual_block(x, name='unit_1_1')

        # unit_2_x
        if self._hp.ngroups3 == 1:
            x = self._residual_block_first(x, filters[2], strides[1], name='unit_2_0')
            x = self._residual_block(x, name='unit_2_1')
        else:
            unit_2_0_shortcut_kernel = self._hp.split_params['unit_2_0']['shortcut']
            unit_2_0_conv1_kernel = self._hp.split_params['unit_2_0']['conv1']
            unit_2_0_conv2_kernel = self._hp.split_params['unit_2_0']['conv2']
            unit_2_0_p_perms = self._hp.split_params['unit_2_0']['p_perms']
            unit_2_0_q_perms = self._hp.split_params['unit_2_0']['q_perms']
            unit_2_0_r_perms = self._hp.split_params['unit_2_0']['r_perms']

            with tf.variable_scope('unit_2_0'):
                shortcut = self._conv_split(x, filters[2], strides[1], unit_2_0_shortcut_kernel, unit_2_0_p_perms, unit_2_0_q_perms, name='shortcut')
                x = self._conv_split(x, filters[2], strides[1], unit_2_0_conv1_kernel, unit_2_0_p_perms, unit_2_0_r_perms, name='conv_1')
                x = self._bn(x, name='bn_1')
                x = self._relu(x, name='relu_1')
                x = self._conv_split(x, filters[2], 1, unit_2_0_conv2_kernel, unit_2_0_r_perms, unit_2_0_q_perms, name='conv_2')
                x = self._bn(x, name='bn_2')
                x = x + shortcut
                x = self._relu(x, name='relu_2')

            unit_2_1_conv1_kernel = self._hp.split_params['unit_2_1']['conv1']
            unit_2_1_conv2_kernel = self._hp.split_params['unit_2_1']['conv2']
            unit_2_1_p_perms = self._hp.split_params['unit_2_1']['p_perms']
            unit_2_1_r_perms = self._hp.split_params['unit_2_1']['r_perms']

            with tf.variable_scope('unit_2_1'):
                shortcut = x
                x = self._conv_split(x, filters[2], 1, unit_2_1_conv1_kernel, unit_2_1_p_perms, unit_2_1_r_perms, name='conv_1')
                x = self._bn(x, name='bn_1')
                x = self._relu(x, name='relu_1')
                x = self._conv_split(x, filters[2], 1, unit_2_1_conv2_kernel, unit_2_1_r_perms, unit_2_1_p_perms, name='conv_2')
                x = self._bn(x, name='bn_2')
                x = x + shortcut
                x = self._relu(x, name='relu_2')

        # unit_3_x
        if self._hp.ngroups2 == 1:
            x = self._residual_block_first(x, filters[3], strides[2], name='unit_3_0')
            x = self._residual_block(x, name='unit_3_1')
        else:
            unit_3_0_shortcut_kernel = self._hp.split_params['unit_3_0']['shortcut']
            unit_3_0_conv1_kernel = self._hp.split_params['unit_3_0']['conv1']
            unit_3_0_conv2_kernel = self._hp.split_params['unit_3_0']['conv2']
            unit_3_0_p_perms = self._hp.split_params['unit_3_0']['p_perms']
            unit_3_0_q_perms = self._hp.split_params['unit_3_0']['q_perms']
            unit_3_0_r_perms = self._hp.split_params['unit_3_0']['r_perms']

            with tf.variable_scope('unit_3_0'):
                shortcut = self._conv_split(x, filters[3], strides[2], unit_3_0_shortcut_kernel, unit_3_0_p_perms, unit_3_0_q_perms, name='shortcut')
                x = self._conv_split(x, filters[3], strides[2], unit_3_0_conv1_kernel, unit_3_0_p_perms, unit_3_0_r_perms, name='conv_1')
                x = self._bn(x, name='bn_1')
                x = self._relu(x, name='relu_1')
                x = self._conv_split(x, filters[3], 1, unit_3_0_conv2_kernel, unit_3_0_r_perms, unit_3_0_q_perms, name='conv_2')
                x = self._bn(x, name='bn_2')
                x = x + shortcut
                x = self._relu(x, name='relu_2')

            unit_3_1_conv1_kernel = self._hp.split_params['unit_3_1']['conv1']
            unit_3_1_conv2_kernel = self._hp.split_params['unit_3_1']['conv2']
            unit_3_1_p_perms = self._hp.split_params['unit_3_1']['p_perms']
            unit_3_1_r_perms = self._hp.split_params['unit_3_1']['r_perms']

            with tf.variable_scope('unit_3_1'):
                shortcut = x
                x = self._conv_split(x, filters[3], 1, unit_3_1_conv1_kernel, unit_3_1_p_perms, unit_3_1_r_perms, name='conv_1')
                x = self._bn(x, name='bn_1')
                x = self._relu(x, name='relu_1')
                x = self._conv_split(x, filters[3], 1, unit_3_1_conv2_kernel, unit_3_1_r_perms, unit_3_1_p_perms, name='conv_2')
                x = self._bn(x, name='bn_2')
                x = x + shortcut
                x = self._relu(x, name='relu_2')

        # Last unit
        with tf.variable_scope('unit_last') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = utils._bn(x, self.is_train, self._global_step)
            x = utils._relu(x)
            x = tf.reduce_mean(x, [1, 2])

        # Logit
        logits_weights = self._hp.split_params['logits']['weights']
        logits_biases = self._hp.split_params['logits']['biases']
        logits_input_perms = self._hp.split_params['logits']['input_perms']
        logits_output_perms = self._hp.split_params['logits']['output_perms']
        with tf.variable_scope('logits') as scope:
            print('\tBuilding unit: %s - %d split' % (scope.name, len(logits_weights)))
            x_offset = 0
            x_list = []
            for i, (w, b, p) in enumerate(zip(logits_weights, logits_biases, logits_input_perms)):
                in_dim, out_dim = w.shape
                x_split = tf.transpose(tf.gather(tf.transpose(x), p))
                x_split = self._fc_with_init(x_split, out_dim, init_w=w, init_b=b, name='split%d' % (i+1))
                x_list.append(x_split)
                x_offset += in_dim
            x = tf.concat(x_list, 1)
            output_forward_idx = list(np.concatenate(logits_output_perms))
            output_inverse_idx = [output_forward_idx.index(i) for i in range(self._hp.num_classes)]
            x = tf.transpose(tf.gather(tf.transpose(x), output_inverse_idx))

        self._logits = x

        # Probs & preds & acc
        self.probs = tf.nn.softmax(x, name='probs')
        self.preds = tf.to_int32(tf.argmax(self._logits, 1, name='preds'))
        ones = tf.constant(np.ones([self._hp.batch_size]), dtype=tf.float32)
        zeros = tf.constant(np.zeros([self._hp.batch_size]), dtype=tf.float32)
        correct = tf.where(tf.equal(self.preds, self._labels), ones, zeros)
        self.acc = tf.reduce_mean(correct, name='acc')
        tf.summary.scalar('accuracy', self.acc)

        # Loss & acc
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=self._labels)
        self.loss = tf.reduce_mean(loss)
        tf.summary.scalar('cross_entropy', self.loss)


    def build_train_op(self):
        print('Building train ops')

        # Learning rate
        tf.summary.scalar('learing_rate', self.lr)

        losses = [self.loss]

        # Add l2 loss
        with tf.variable_scope('l2_loss'):
            costs = [tf.nn.l2_loss(var) for var in tf.get_collection(utils.WEIGHT_DECAY_KEY)]
            l2_loss = tf.multiply(self._hp.weight_decay, tf.add_n(costs))
            losses.append(l2_loss)

        self._total_loss = tf.add_n(losses)

        # Gradient descent step
        opt = tf.train.MomentumOptimizer(self.lr, self._hp.momentum)
        grads_and_vars = opt.compute_gradients(self._total_loss, tf.trainable_variables())
        if self._hp.finetune:
            for idx, (grad, var) in enumerate(grads_and_vars):
                if "group" in var.op.name or \
                        (("unit_1_0" in var.op.name or "unit_1_1" in var.op.name) and self._hp.ngroups3 > 1) or \
                        (("unit_2_0" in var.op.name or "unit_2_1" in var.op.name) and self._hp.ngroups2 > 1) or \
                        ("unit_3_0" in var.op.name or "unit_3_1" in var.op.name) or \
                        "logits" in var.op.name:
                    print('\tScale up learning rate of % s by 10.0' % var.op.name)
                    grad = 10.0 * grad
                    grads_and_vars[idx] = (grad,var)

        # Apply gradient
        apply_grad_op = opt.apply_gradients(grads_and_vars, global_step=self._global_step)

        # Batch normalization moving average update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            with tf.control_dependencies(update_ops+[apply_grad_op]):
                self.train_op = tf.no_op()
        else:
            self.train_op = apply_grad_op


    def _residual_block_first(self, x, out_channel, strides, input_q=None, output_q=None, split_r=None, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            if input_q is not None and output_q is not None and split_r is not None:
                x = self._dropout(x, self._hp.dropout_keep_prob, name='dropout_1')
            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, out_channel, strides, input_q=input_q, output_q=output_q, name='shortcut')
            # Residual
            x = self._conv(x, 3, out_channel, strides, input_q=input_q, output_q=split_r, name='conv_1')
            x = self._bn(x, name='bn_2')
            x = self._relu(x, name='relu_2')
            if input_q is not None and output_q is not None and split_r is not None:
                x = self._dropout(x, self._hp.dropout_keep_prob, name='dropout_2')
            x = self._conv(x, 3, out_channel, 1, input_q=split_r, output_q=output_q, name='conv_2')
            # Merge
            x = x + shortcut
        return x

    def _residual_block(self, x, split_q=None, split_r=None, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut connection
            shortcut = x
            # Residual
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            if split_q is not None and split_r is not None:
                x = self._dropout(x, self._hp.dropout_keep_prob, name='dropout_1')
            x = self._conv(x, 3, num_channel, 1, input_q=split_q, output_q=split_r, name='conv_1')
            x = self._bn(x, name='bn_2')
            x = self._relu(x, name='relu_2')
            if split_q is not None and split_r is not None:
                x = self._dropout(x, self._hp.dropout_keep_prob, name='dropout_2')
            x = self._conv(x, 3, num_channel, 1, input_q=split_r, output_q=split_q, name='conv_2')
            # Merge
            x = x + shortcut
        return x


    def _conv_split(self, x, out_channel, strides, kernels, input_perms, output_perms, name="unit"):
        b, w, h, in_channel = x.get_shape().as_list()
        x_list = []
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s - %d split' % (scope.name, len(kernels)))
            for i, (k, p) in enumerate(zip(kernels, input_perms)):
                kernel_size, in_dim, out_dim = k.shape[-3:]
                x_split = tf.transpose(tf.gather(tf.transpose(x, (3, 0, 1, 2)), p), (1, 2, 3, 0))
                x_split = self._conv_with_init(x_split, kernel_size, out_dim, strides, init_k=k, name="split%d"%(i+1))
                x_list.append(x_split)
        x = tf.concat(x_list, 3)
        output_forward_idx = list(np.concatenate(output_perms))
        output_inverse_idx = [output_forward_idx.index(i) for i in range(out_channel)]
        x = tf.transpose(tf.gather(tf.transpose(x, (3, 0, 1, 2)), output_inverse_idx), (1, 2, 3, 0))
        return x


    # Helper functions(counts FLOPs and number of weights)
    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", input_q=None, output_q=None, name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._conv(x, filter_size, out_channel, stride, pad, input_q, output_q, name)
        f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _conv_with_init(self, x, filter_size, out_channel, stride, pad="SAME", init_k=None, name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._conv_with_init(x, filter_size, out_channel, stride, pad, init_k, name)
        f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _fc(self, x, out_dim, input_q=None, output_q=None, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = utils._fc(x, out_dim, input_q, output_q, name)
        f = 2 * (in_dim + 1) * out_dim
        w = (in_dim + 1) * out_dim
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _fc_with_init(self, x, out_dim, init_w=None, init_b=None, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = utils._fc_with_init(x, out_dim, init_w, init_b, name)
        f = 2*(in_dim + 1) * out_dim
        w = (in_dim + 1) * out_dim
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _bn(self, x, name="bn"):
        x = utils._bn(x, self.is_train, self._global_step, name)
        # f = 8 * self._get_data_size(x)
        # w = 4 * x.get_shape().as_list()[-1]
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, w)
        return x

    def _relu(self, x, name="relu"):
        x = utils._relu(x, 0.0, name)
        # f = self._get_data_size(x)
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, 0)
        return x

    def _dropout(self, x, keep_prob, name="dropout"):
        x = utils._dropout(x, keep_prob, name)
        return x

    def _get_data_size(self, x):
        return np.prod(x.get_shape().as_list()[1:])

    def _add_flops_weights(self, scope_name, f, w):
        if scope_name not in self._counted_scope:
            self._flops += f
            self._weights += w
            self._counted_scope.append(scope_name)
