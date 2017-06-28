#!/usr/bin/env python

import os
from datetime import datetime
import time
import tensorflow as tf
import numpy as np
import sys
import select
from IPython import embed
from StringIO import StringIO
import matplotlib.pyplot as plt

import cifar100
import resnet_split as resnet



# Dataset Configuration
tf.app.flags.DEFINE_string('data_dir', './cifar100/train_val_split', """Path to the CIFAR-100 data.""")
tf.app.flags.DEFINE_integer('num_classes', 100, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_train_instance', 45000, """Number of training images.""")
tf.app.flags.DEFINE_integer('num_val_instance', 5000, """Number of val images.""")

# Network Configuration
tf.app.flags.DEFINE_integer('batch_size', 90, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_residual_units', 2, """Number of residual block per group.
                                                Total number of conv layers will be 6n+4""")
tf.app.flags.DEFINE_integer('k', 8, """Network width multiplier""")
tf.app.flags.DEFINE_integer('ngroups1', 1, """Grouping number on logits""")
tf.app.flags.DEFINE_integer('ngroups2', 1, """Grouping number on unit_3_x""")
tf.app.flags.DEFINE_integer('ngroups3', 1, """Grouping number on unit_2_x""")

# Optimization Configuration
tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
tf.app.flags.DEFINE_string('lr_step_epoch', "80.0,120.0,160.0", """Epochs after which learing rate decays""")
tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")
tf.app.flags.DEFINE_boolean('finetune', False, """Whether to finetune.""")

# Training Configuration
tf.app.flags.DEFINE_string('train_dir', './train', """Directory where to write log and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_integer('val_interval', 1000, """Number of iterations to run a val""")
tf.app.flags.DEFINE_integer('val_iter', 100, """Number of iterations during a val""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 10000, """Number of iterations to save parameters as a checkpoint""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_string('basemodel', None, """Base model to load paramters""")
tf.app.flags.DEFINE_string('checkpoint', None, """Model checkpoint to load""")

FLAGS = tf.app.flags.FLAGS


def get_lr(initial_lr, lr_decay, lr_decay_steps, global_step):
    lr = initial_lr
    for s in lr_decay_steps:
        if global_step >= s:
            lr *= lr_decay
    return lr


def train():
    print('[Dataset Configuration]')
    print('\tCIFAR-100 dir: %s' % FLAGS.data_dir)
    print('\tNumber of classes: %d' % FLAGS.num_classes)
    print('\tNumber of training images: %d' % FLAGS.num_train_instance)
    print('\tNumber of val images: %d' % FLAGS.num_val_instance)

    print('[Network Configuration]')
    print('\tBatch size: %d' % FLAGS.batch_size)
    print('\tResidual blocks per group: %d' % FLAGS.num_residual_units)
    print('\tNetwork width multiplier: %d' % FLAGS.k)
    print('\tNumber of Groups: %d-%d-%d' % (FLAGS.ngroups3, FLAGS.ngroups2, FLAGS.ngroups1))
    print('\tBasemodel file: %s' % FLAGS.basemodel)

    print('[Optimization Configuration]')
    print('\tL2 loss weight: %f' % FLAGS.l2_weight)
    print('\tThe momentum optimizer: %f' % FLAGS.momentum)
    print('\tInitial learning rate: %f' % FLAGS.initial_lr)
    print('\tEpochs per lr step: %s' % FLAGS.lr_step_epoch)
    print('\tLearning rate decay: %f' % FLAGS.lr_decay)
    print('\tFinetune: %d' % FLAGS.finetune)

    print('[Training Configuration]')
    print('\tTrain dir: %s' % FLAGS.train_dir)
    print('\tTraining max steps: %d' % FLAGS.max_steps)
    print('\tSteps per displaying info: %d' % FLAGS.display)
    print('\tSteps per validation: %d' % FLAGS.val_interval)
    print('\tSteps during validation: %d' % FLAGS.val_iter)
    print('\tSteps per saving checkpoints: %d' % FLAGS.checkpoint_interval)
    print('\tGPU memory fraction: %f' % FLAGS.gpu_fraction)
    print('\tLog device placement: %d' % FLAGS.log_device_placement)


    with tf.Graph().as_default():
        init_step = 0
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get images and labels of CIFAR-100
        print('Load CIFAR-100 dataset')
        train_dataset_path = os.path.join(FLAGS.data_dir, 'train')
        val_dataset_path = os.path.join(FLAGS.data_dir, 'val')
        print('\tLoading training data from %s' % train_dataset_path)
        with tf.variable_scope('train_image'):
            cifar100_train = cifar100.CIFAR100Runner(train_dataset_path, image_per_thread=32,
                                    shuffle=True, distort=True, capacity=10000)
            train_images, train_labels = cifar100_train.get_inputs(FLAGS.batch_size)
        print('\tLoading validation data from %s' % val_dataset_path)
        with tf.variable_scope('val_image'):
            cifar100_val = cifar100.CIFAR100Runner(val_dataset_path, image_per_thread=32,
                                    shuffle=False, distort=False, capacity=5000)
                                    # shuffle=False, distort=False, capacity=10000)
            val_images, val_labels = cifar100_val.get_inputs(FLAGS.batch_size)

        # Build a Graph that computes the predictions from the inference model.
        images = tf.placeholder(tf.float32, [FLAGS.batch_size, cifar100.IMAGE_SIZE, cifar100.IMAGE_SIZE, 3])
        labels = tf.placeholder(tf.int32, [FLAGS.batch_size])

        # Get splitted params
        if not FLAGS.basemodel:
            print('No basemodel found to load split params')
            sys.exit(-1)
        else:
            print('Load split params from %s' % FLAGS.basemodel)

            def get_perms(q_name, ngroups):
                split_alpha = reader.get_tensor(q_name+'/alpha')
                q_amax = np.argmax(split_alpha, axis=0)
                return [np.where(q_amax == i)[0] for i in range(ngroups)]

            reader = tf.train.NewCheckpointReader(FLAGS.basemodel)
            split_params = {}

            print('\tlogits...')
            base_logits_w = reader.get_tensor('logits/fc/weights')
            base_logits_b = reader.get_tensor('logits/fc/biases')
            split_p1_idxs = get_perms('group/split_p1', FLAGS.ngroups1)
            split_q1_idxs = get_perms('group/split_q1', FLAGS.ngroups1)

            logits_params = {'weights':[], 'biases':[], 'input_perms':[], 'output_perms':[]}
            for i in range(FLAGS.ngroups1):
                logits_params['weights'].append(base_logits_w[split_p1_idxs[i], :][:, split_q1_idxs[i]])
                logits_params['biases'].append(base_logits_b[split_q1_idxs[i]])
            logits_params['input_perms'] = split_p1_idxs
            logits_params['output_perms'] = split_q1_idxs
            split_params['logits'] = logits_params

            if FLAGS.ngroups2 > 1:
                print('\tunit_3_x...')
                base_unit_3_0_shortcut_k = reader.get_tensor('unit_3_0/shortcut/kernel')
                base_unit_3_0_conv1_k = reader.get_tensor('unit_3_0/conv_1/kernel')
                base_unit_3_0_conv2_k = reader.get_tensor('unit_3_0/conv_2/kernel')
                base_unit_3_1_conv1_k = reader.get_tensor('unit_3_1/conv_1/kernel')
                base_unit_3_1_conv2_k = reader.get_tensor('unit_3_1/conv_2/kernel')
                split_p2_idxs = get_perms('group/split_p2', FLAGS.ngroups2)
                split_q2_idxs = _merge_split_idxs(split_p1_idxs, _get_even_merge_idxs(FLAGS.ngroups1, FLAGS.ngroups2))
                split_r21_idxs = get_perms('group/split_r21', FLAGS.ngroups2)
                split_r22_idxs = get_perms('group/split_r22', FLAGS.ngroups2)

                unit_3_0_params = {'shortcut':[], 'conv1':[], 'conv2':[], 'p_perms':[], 'q_perms':[], 'r_perms':[]}
                for i in range(FLAGS.ngroups2):
                    unit_3_0_params['shortcut'].append(base_unit_3_0_shortcut_k[:,:,split_p2_idxs[i],:][:,:,:,split_q2_idxs[i]])
                    unit_3_0_params['conv1'].append(base_unit_3_0_conv1_k[:,:,split_p2_idxs[i],:][:,:,:,split_r21_idxs[i]])
                    unit_3_0_params['conv2'].append(base_unit_3_0_conv2_k[:,:,split_r21_idxs[i],:][:,:,:,split_q2_idxs[i]])
                unit_3_0_params['p_perms'] = split_p2_idxs
                unit_3_0_params['q_perms'] = split_q2_idxs
                unit_3_0_params['r_perms'] = split_r21_idxs
                split_params['unit_3_0'] = unit_3_0_params

                unit_3_1_params = {'conv1':[], 'conv2':[], 'p_perms':[], 'r_perms':[]}
                for i in range(FLAGS.ngroups2):
                    unit_3_1_params['conv1'].append(base_unit_3_1_conv1_k[:,:,split_q2_idxs[i],:][:,:,:,split_r22_idxs[i]])
                    unit_3_1_params['conv2'].append(base_unit_3_1_conv2_k[:,:,split_r22_idxs[i],:][:,:,:,split_q2_idxs[i]])
                unit_3_1_params['p_perms'] = split_q2_idxs
                unit_3_1_params['r_perms'] = split_r22_idxs
                split_params['unit_3_1'] = unit_3_1_params

            if FLAGS.ngroups3 > 1:
                print('\tconv4_x...')
                base_unit_2_0_shortcut_k = reader.get_tensor('unit_2_0/shortcut/kernel')
                base_unit_2_0_conv1_k = reader.get_tensor('unit_2_0/conv_1/kernel')
                base_unit_2_0_conv2_k = reader.get_tensor('unit_2_0/conv_2/kernel')
                base_unit_2_1_conv1_k = reader.get_tensor('unit_2_1/conv_1/kernel')
                base_unit_2_1_conv2_k = reader.get_tensor('unit_2_1/conv_2/kernel')
                split_p3_idxs = get_perms('group/split_p3', FLAGS.ngroups3)
                split_q3_idxs = _merge_split_idxs(split_p2_idxs, _get_even_merge_idxs(FLAGS.ngroups2, FLAGS.ngroups3))
                split_r31_idxs = get_perms('group/split_r31', FLAGS.ngroups3)
                split_r32_idxs = get_perms('group/split_r32', FLAGS.ngroups3)

                unit_2_0_params = {'shortcut':[], 'conv1':[], 'conv2':[], 'p_perms':[], 'q_perms':[], 'r_perms':[]}
                for i in range(FLAGS.ngroups3):
                    unit_2_0_params['shortcut'].append(base_unit_2_0_shortcut_k[:,:,split_p3_idxs[i],:][:,:,:,split_q3_idxs[i]])
                    unit_2_0_params['conv1'].append(base_unit_2_0_conv1_k[:,:,split_p3_idxs[i],:][:,:,:,split_r31_idxs[i]])
                    unit_2_0_params['conv2'].append(base_unit_2_0_conv2_k[:,:,split_r31_idxs[i],:][:,:,:,split_q3_idxs[i]])
                unit_2_0_params['p_perms'] = split_p3_idxs
                unit_2_0_params['q_perms'] = split_q3_idxs
                unit_2_0_params['r_perms'] = split_r31_idxs
                split_params['unit_2_0'] = unit_2_0_params

                unit_2_1_params = {'conv1':[], 'conv2':[], 'p_perms':[], 'r_perms':[]}
                for i in range(FLAGS.ngroups3):
                    unit_2_1_params['conv1'].append(base_unit_2_1_conv1_k[:,:,split_q3_idxs[i],:][:,:,:,split_r32_idxs[i]])
                    unit_2_1_params['conv2'].append(base_unit_2_1_conv2_k[:,:,split_r32_idxs[i],:][:,:,:,split_q3_idxs[i]])
                unit_2_1_params['p_perms'] = split_q3_idxs
                unit_2_1_params['r_perms'] = split_r32_idxs
                split_params['unit_2_1'] = unit_2_1_params


        # Build model
        lr_decay_steps = map(float,FLAGS.lr_step_epoch.split(','))
        lr_decay_steps = map(int,[s*FLAGS.num_train_instance/FLAGS.batch_size for s in lr_decay_steps])
        hp = resnet.HParams(batch_size=FLAGS.batch_size,
                            num_classes=FLAGS.num_classes,
                            num_residual_units=FLAGS.num_residual_units,
                            k=FLAGS.k,
                            weight_decay=FLAGS.l2_weight,
                            ngroups1=FLAGS.ngroups1,
                            ngroups2=FLAGS.ngroups2,
                            ngroups3=FLAGS.ngroups3,
                            split_params=split_params,
                            momentum=FLAGS.momentum,
                            finetune=FLAGS.finetune)
        network = resnet.ResNet(hp, images, labels, global_step)
        network.build_model()
        network.build_train_op()
        print('Number of Weights: %d' % network._weights)
        print('FLOPs: %d' % network._flops)

        train_summary_op = tf.summary.merge_all()  # Summaries(training)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            # allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))

        '''debugging attempt
        from tensorflow.python import debug as tf_debug
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        def _get_data(datum, tensor):
            return tensor == train_images
        sess.add_tensor_filter("get_data", _get_data)
        '''

        sess.run(init)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
        if FLAGS.checkpoint is not None:
            saver.restore(sess, FLAGS.checkpoint)
            init_step = global_step.eval(session=sess)
            print('Load checkpoint %s' % FLAGS.checkpoint)
        else:
            # Define a different saver to load model checkpoints
            # Select only base variables (exclude split layers)
            print('Load parameters from basemodel %s' % FLAGS.basemodel)
            variables = tf.global_variables()
            vars_restore = [var for var in variables
                            if not "Momentum" in var.name and
                               not "logits" in var.name and
                               not "global_step" in var.name]
            if FLAGS.ngroups2 > 1:
                vars_restore = [var for var in vars_restore
                                if not "unit_3_" in var.name]
            if FLAGS.ngroups3 > 1:
                vars_restore = [var for var in vars_restore
                                if not "unit_2_" in var.name]
            saver_restore = tf.train.Saver(vars_restore, max_to_keep=10000)
            saver_restore.restore(sess, FLAGS.basemodel)

        # Start queue runners & summary_writer
        cifar100_train.start_threads(sess, n_threads=20)
        cifar100_val.start_threads(sess, n_threads=1)

        if not os.path.exists(FLAGS.train_dir):
            os.mkdir(FLAGS.train_dir)
        summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, str(global_step.eval(session=sess))),
                                               sess.graph)

        # Training!
        val_best_acc = 0.0
        for step in xrange(init_step, FLAGS.max_steps):
            # val
            if step % FLAGS.val_interval == 0:
                val_loss, val_acc = 0.0, 0.0
                for i in range(FLAGS.val_iter):
                    val_images_val, val_labels_val = sess.run([val_images, val_labels])
                    loss_value, acc_value = sess.run([network.loss, network.acc],
                                feed_dict={network.is_train:False, images:val_images_val, labels:val_labels_val})
                    val_loss += loss_value
                    val_acc += acc_value
                val_loss /= FLAGS.val_iter
                val_acc /= FLAGS.val_iter
                val_best_acc = max(val_best_acc, val_acc)
                format_str = ('%s: (val)     step %d, loss=%.4f, acc=%.4f')
                print (format_str % (datetime.now(), step, val_loss, val_acc))

                val_summary = tf.Summary()
                val_summary.value.add(tag='val/loss', simple_value=val_loss)
                val_summary.value.add(tag='val/acc', simple_value=val_acc)
                val_summary.value.add(tag='val/best_acc', simple_value=val_best_acc)
                summary_writer.add_summary(val_summary, step)
                summary_writer.flush()

            # Train
            lr_value = get_lr(FLAGS.initial_lr, FLAGS.lr_decay, lr_decay_steps, step)
            start_time = time.time()
            train_images_val, train_labels_val = sess.run([train_images, train_labels])
            _, loss_value, acc_value, train_summary_str = \
                    sess.run([network.train_op, network.loss, network.acc, train_summary_op],
                             feed_dict={network.is_train:True, network.lr:lr_value, images:train_images_val, labels:train_labels_val})
            duration = time.time() - start_time

            assert not np.isnan(loss_value)

            # Display & Summary(training)
            if step % FLAGS.display == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: (Training) step %d, loss=%.4f, acc=%.4f, lr=%f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value, acc_value, lr_value,
                                     examples_per_sec, sec_per_batch))
                summary_writer.add_summary(train_summary_str, step)

            # Save the model checkpoint periodically.
            if (step > init_step and step % FLAGS.checkpoint_interval == 0) or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
              char = sys.stdin.read(1)
              if char == 'b':
                embed()


def _merge_split_q(q, merge_idxs, name='merge'):
    ngroups, dim = q.shape
    max_idx = np.max(merge_idxs)
    temp_list = []
    for i in range(max_idx + 1):
        temp = []
        for j in range(ngroups):
            if merge_idxs[j] == i:
                 temp.append(q[j,:])
        temp_list.append(np.sum(temp, axis=0))
    ret = np.array(temp_list)

    return ret

def _merge_split_idxs(split_idxs, merge_idxs, name='merge'):
    ngroups = len(split_idxs)
    max_idx = np.max(merge_idxs)
    ret = []
    for i in range(max_idx + 1):
        temp = []
        for j in range(ngroups):
            if merge_idxs[j] == i:
                 temp.append(split_idxs[j])
        ret.append(np.concatenate(temp))

    return ret

def _get_even_merge_idxs(N, split):
    assert N >= split
    num_elems = [(N + split - i - 1)/split for i in range(split)]
    expand_split = [[i] * n for i, n in enumerate(num_elems)]
    return [t for l in expand_split for t in l]


def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
