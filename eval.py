#!/usr/bin/env python

import os
import sys
from datetime import datetime
import time
import tensorflow as tf
import numpy as np
import cPickle as pickle

import cifar100
import resnet_split as resnet



# Dataset Configuration
tf.app.flags.DEFINE_string('data_dir', './cifar100/train_val_split', """Path to the CIFAR-100 data.""")
tf.app.flags.DEFINE_integer('num_classes', 100, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_test_instance', 10000, """Number of test images.""")

# Network Configuration
tf.app.flags.DEFINE_integer('batch_size', 100, """Number of images to process in a batch.""")
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

# Evaluation Configuration
tf.app.flags.DEFINE_string('basemodel', './group/model.ckpt-199999', """Base model to load paramters""")
tf.app.flags.DEFINE_string('checkpoint', './split/model.ckpt-149999', """Path to the model checkpoint file""")
tf.app.flags.DEFINE_string('output_file', './split/eval.pkl', """Path to the result pkl file""")
tf.app.flags.DEFINE_integer('test_iter', 100, """Number of test batches during the evaluation""")
tf.app.flags.DEFINE_integer('display', 10, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

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
    print('\tNumber of test images: %d' % FLAGS.num_test_instance)

    print('[Network Configuration]')
    print('\tBatch size: %d' % FLAGS.batch_size)
    print('\tResidual blocks per group: %d' % FLAGS.num_residual_units)
    print('\tNetwork width multiplier: %d' % FLAGS.k)
    print('\tNumber of Groups: %d-%d-%d' % (FLAGS.ngroups3, FLAGS.ngroups2, FLAGS.ngroups1))
    print('\tBasemodel file: %s' % FLAGS.basemodel)

    print('[Evaluation Configuration]')
    print('\tCheckpoint file: %s' % FLAGS.checkpoint)
    print('\tOutput file path: %s' % FLAGS.output_file)
    print('\tTest iterations: %d' % FLAGS.test_iter)
    print('\tSteps per displaying info: %d' % FLAGS.display)
    print('\tGPU memory fraction: %f' % FLAGS.gpu_fraction)
    print('\tLog device placement: %d' % FLAGS.log_device_placement)


    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get images and labels of CIFAR-100
        print('Load CIFAR-100 dataset')
        test_dataset_path = os.path.join(FLAGS.data_dir, 'test')
        with tf.variable_scope('test_image'):
            cifar100_test = cifar100.CIFAR100Runner(test_dataset_path, image_per_thread=1,
                                    shuffle=False, distort=False, capacity=5000)
            test_images, test_labels = cifar100_test.get_inputs(FLAGS.batch_size)

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
        print('Number of Weights: %d' % network._weights)
        print('FLOPs: %d' % network._flops)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            allow_soft_placement=True,
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
           print('Load checkpoint %s' % FLAGS.checkpoint)
        else:
            print('No checkpoint file found.')
            sys.exit(1)

        # Start queue runners & summary_writer
        cifar100_test.start_threads(sess, n_threads=1)

        # Test!
        test_loss = 0.0
        test_acc = 0.0
        test_time = 0.0
        confusion_matrix = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.int32)
        for i in range(FLAGS.test_iter):
            test_images_val, test_labels_val = sess.run([test_images, test_labels])
            start_time = time.time()
            loss_value, acc_value, pred_value = sess.run([network.loss, network.acc, network.preds],
                        feed_dict={network.is_train:False, images:test_images_val, labels:test_labels_val})
            duration = time.time() - start_time
            test_loss += loss_value
            test_acc += acc_value
            test_time += duration
            for l, p in zip(test_labels_val, pred_value):
                confusion_matrix[l, p] += 1

            if i % FLAGS.display == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: iter %d, loss=%.4f, acc=%.4f (%.1f examples/sec; %.3f sec/batch)')
                print (format_str % (datetime.now(), i, loss_value, acc_value,
                                     examples_per_sec, sec_per_batch))
        test_loss /= FLAGS.test_iter
        test_acc /= FLAGS.test_iter

        # Print and save results
        sec_per_image = test_time/FLAGS.test_iter/FLAGS.batch_size
        print ('Done! Acc: %.6f, Test time: %.3f sec, %.7f sec/example' % (test_acc, test_time, sec_per_image))
        print ('Saving result... ')
        result = {'accuracy': test_acc, 'confusion_matrix': confusion_matrix,
                  'test_time': test_time, 'sec_per_image': sec_per_image}
        with open(FLAGS.output_file, 'wb') as fd:
            pickle.dump(result, fd)
        print ('done!')


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
