#!/usr/bin/env python

from collections import namedtuple

import tensorflow as tf
import numpy as np

import utils


HParams = namedtuple('HParams',
                    'batch_size, num_classes, num_residual_units, k, weight_decay, momentum, finetune, '
                    'ngroups1, ngroups2, ngroups3, gamma1, gamma2, gamma3, '
                    'dropout_keep_prob, bn_no_scale, weighted_group_loss')

class ResNet(object):
    def __init__(self, hp, images, labels, global_step):
        self._hp = hp # Hyperparameters
        self._images = images # Input image
        self._labels = labels
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

        with tf.variable_scope("group"):
            if self._hp.ngroups1 > 1:
                self.split_q1 = utils._get_split_q(self._hp.ngroups1, self._hp.num_classes, name='split_q1')
                self.split_p1 = utils._get_split_q(self._hp.ngroups1, filters[3], name='split_p1')
                tf.summary.histogram("group/split_p1/", self.split_p1)
                tf.summary.histogram("group/split_q1/", self.split_q1)
            else:
                self.split_q1 = None
                self.split_p1 = None

            if self._hp.ngroups2 > 1:
                self.split_q2 = utils._merge_split_q(self.split_p1, utils._get_even_merge_idxs(self._hp.ngroups1, self._hp.ngroups2), name='split_q2')
                self.split_p2 = utils._get_split_q(self._hp.ngroups2, filters[2], name='split_p2')
                self.split_r21 = utils._get_split_q(self._hp.ngroups2, filters[3], name='split_r21')
                self.split_r22 = utils._get_split_q(self._hp.ngroups2, filters[3], name='split_r22')
                tf.summary.histogram("group/split_q2/", self.split_q2)
                tf.summary.histogram("group/split_p2/", self.split_p2)
                tf.summary.histogram("group/split_r21/", self.split_r21)
                tf.summary.histogram("group/split_r22/", self.split_r22)
            else:
                self.split_p2 = None
                self.split_q2 = None
                self.split_r21 = None
                self.split_r22 = None

            if self._hp.ngroups3 > 1:
                self.split_q3 = utils._merge_split_q(self.split_p2, utils._get_even_merge_idxs(self._hp.ngroups2, self._hp.ngroups3), name='split_q3')
                self.split_p3 = utils._get_split_q(self._hp.ngroups3, filters[1], name='split_p3')
                self.split_r31 = utils._get_split_q(self._hp.ngroups3, filters[2], name='split_r31')
                self.split_r32 = utils._get_split_q(self._hp.ngroups3, filters[2], name='split_r32')
                tf.summary.histogram("group/split_q3/", self.split_q3)
                tf.summary.histogram("group/split_p3/", self.split_p3)
                tf.summary.histogram("group/split_r31/", self.split_r31)
                tf.summary.histogram("group/split_r32/", self.split_r32)
            else:
                self.split_p3 = None
                self.split_q3 = None
                self.split_r31 = None
                self.split_r32 = None

        # Init. conv.
        print('\tBuilding unit: init_conv')
        x = utils._conv(self._images, 3, filters[0], 1, name='init_conv')

        x = self._residual_block_first(x, filters[1], strides[0], name='unit_1_0')
        x = self._residual_block(x, name='unit_1_1')

        x = self._residual_block_first(x, filters[2], strides[1], input_q=self.split_p3, output_q=self.split_q3, split_r=self.split_r31, name='unit_2_0')
        x = self._residual_block(x, split_q=self.split_q3, split_r=self.split_r32, name='unit_2_1')

        x = self._residual_block_first(x, filters[3], strides[2], input_q=self.split_p2, output_q=self.split_q2, split_r=self.split_r21, name='unit_3_0')
        x = self._residual_block(x, split_q=self.split_q2, split_r=self.split_r22, name='unit_3_1')

        # Last unit
        with tf.variable_scope('unit_last') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = utils._bn(x, self.is_train, self._global_step)
            x = utils._relu(x)
            x = tf.reduce_mean(x, [1, 2])

        # Logit
        with tf.variable_scope('logits') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x_shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, x_shape[1]])
            if self.split_p1 is not None and self.split_q1 is not None:
                x = self._dropout(x, self._hp.dropout_keep_prob, name='dropout')
            x = self._fc(x, self._hp.num_classes, input_q=self.split_p1, output_q=self.split_q1)

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


    def _residual_block_first(self, x, out_channel, strides, input_q=None, output_q=None, split_r=None, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            x = self._bn(x, name='bn_1', no_scale=self._hp.bn_no_scale)
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
            x = self._bn(x, name='bn_2', no_scale=self._hp.bn_no_scale)
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
            x = self._bn(x, name='bn_1', no_scale=self._hp.bn_no_scale)
            x = self._relu(x, name='relu_1')
            if split_q is not None and split_r is not None:
                x = self._dropout(x, self._hp.dropout_keep_prob, name='dropout_1')
            x = self._conv(x, 3, num_channel, 1, input_q=split_q, output_q=split_r, name='conv_1')
            x = self._bn(x, name='bn_2', no_scale=self._hp.bn_no_scale)
            x = self._relu(x, name='relu_2')
            if split_q is not None and split_r is not None:
                x = self._dropout(x, self._hp.dropout_keep_prob, name='dropout_2')
            x = self._conv(x, 3, num_channel, 1, input_q=split_r, output_q=split_q, name='conv_2')
            # Merge
            x = x + shortcut
        return x

    def build_train_op(self):
        # Learning rate
        tf.summary.scalar('learing_rate', self.lr)

        losses = [self.loss]

        # Add l2 loss
        with tf.variable_scope('l2_loss'):
            costs = [tf.nn.l2_loss(var) for var in tf.get_collection(utils.WEIGHT_DECAY_KEY)]
            l2_loss = tf.multiply(self._hp.weight_decay, tf.add_n(costs))
            losses.append(l2_loss)

        # Add group split loss
        with tf.variable_scope('group/'):
            if tf.get_collection('OVERLAP_LOSS') and self._hp.gamma1 > 0:
                cost1 = tf.reduce_mean(tf.get_collection('OVERLAP_LOSS'))
                cost1 = cost1 * self._hp.gamma1
                tf.summary.scalar('group/overlap_loss/', cost1)
                losses.append(cost1)

            if tf.get_collection('WEIGHT_SPLIT') and self._hp.gamma2 > 0:
                if self._hp.weighted_group_loss:
                    reg_weights = [tf.stop_gradient(x) for x in tf.get_collection('WEIGHT_SPLIT')]
                    regs = [tf.stop_gradient(x) * x for x in tf.get_collection('WEIGHT_SPLIT')]
                    cost2 = tf.reduce_sum(regs) / tf.reduce_sum(reg_weights)
                else:
                    cost2 = tf.reduce_mean(tf.get_collection('WEIGHT_SPLIT'))
                cost2 = cost2 * self._hp.gamma2
                tf.summary.scalar('group/weight_split_loss/', cost2)
                losses.append(cost2)

            if tf.get_collection('UNIFORM_LOSS') and self._hp.gamma3 > 0:
                cost3 = tf.reduce_mean(tf.get_collection('UNIFORM_LOSS'))
                cost3 = cost3 * self._hp.gamma3
                tf.summary.scalar('group/group_uniform_loss/', cost3)
                losses.append(cost3)

        self._total_loss = tf.add_n(losses)

        # Gradient descent step
        opt = tf.train.MomentumOptimizer(self.lr, self._hp.momentum)
        grads_and_vars = opt.compute_gradients(self._total_loss, tf.trainable_variables())
        if self._hp.finetune:
          for idx, (grad, var) in enumerate(grads_and_vars):
            if "unit3" in var.op.name or \
              "unit_last" in var.op.name or \
              "logits" in var.op.name:
              print('Scale up learning rate of % s by 10.0' % var.op.name)
              grad = 10.0 * grad
          grads_and_vars[idx] = (grad,var)

        apply_grad_op = opt.apply_gradients(grads_and_vars, global_step=self._global_step)


        # Batch normalization moving average update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            with tf.control_dependencies(update_ops+[apply_grad_op]):
                self.train_op = tf.no_op()
        else:
            self.train_op = apply_grad_op

    # Helper functions(counts FLOPs and number of weights)
    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", input_q=None, output_q=None, name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._conv(x, filter_size, out_channel, stride, pad, input_q, output_q, name)
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

    def _bn(self, x, name="bn", no_scale=False):
        x = utils._bn(x, self.is_train, self._global_step, name, no_scale=no_scale)
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
