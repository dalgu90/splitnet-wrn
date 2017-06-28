from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import threading
import cPickle as pickle
import numpy as np
import skimage.util

import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 32

# Global constants describing the CIFAR-100 data set.
NUM_CLASSES = 100


class ThreadsafeIter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


class CIFAR100Runner(object):
    _image_summary_added = False
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """
    def __init__(self, pkl_path, shuffle=False, distort=False,
                 capacity=2000, image_per_thread=16):
        self._shuffle = shuffle
        self._distort = distort
        with open(pkl_path, 'rb') as fd:
            data = pickle.load(fd)
        self._images = data['data'].reshape([-1, 3, 32, 32]).transpose((0, 2, 3, 1)).copy(order='C')
        self._labels = data['labels']  # numpy 1-D array
        self.size = len(self._labels)

        self.queue = tf.FIFOQueue(shapes=[[32,32,3], []],
                                  dtypes=[tf.float32, tf.int32],
                                  capacity=capacity)
        # self.queue = tf.RandomShuffleQueue(shapes=[[32,32,3], []],
                                           # dtypes=[tf.float32, tf.int32],
                                           # capacity=capacity,
                                           # min_after_dequeue=min_after_dequeue)
        self.dataX = tf.placeholder(dtype=tf.float32, shape=[None,32,32,3])
        self.dataY = tf.placeholder(dtype=tf.int32, shape=[None,])
        self.enqueue_op = self.queue.enqueue_many([self.dataX, self.dataY])
        self.image_per_thread = image_per_thread

        self._image_summary_added = False


    def _preprocess_image(self, input_image):
        """Preprocess a single image by crop and whitening(and augmenting if needed).

        Args:
            input_image: An image. 3D tensor of [height, width, channel] size.

        Returns:
            output_image: Preprocessed image. 3D tensor of size same as input_image.gj
        """
        # Crop
        image = input_image
        if self._distort:
            image = skimage.util.pad(image, ((4,4), (4,4), (0,0)), 'reflect')
            crop_h = image.shape[0] - 32
            crop_h_before = random.choice(range(crop_h))
            crop_h_after = crop_h - crop_h_before
            crop_w = image.shape[1] - 32
            crop_w_before = random.choice(range(crop_w))
            crop_w_after = crop_w - crop_w_before
            image = skimage.util.crop(image, ((crop_h_before, crop_h_after), (crop_w_before, crop_w_after), (0, 0)))
        else:
            crop_h = image.shape[0] - 32
            crop_w = image.shape[1] - 32
            if crop_w != 0 or crop_h != 0:
                image = skimage.util.crop(image, ((crop_h/2, (crop_h+1)/2), (crop_w/2, (crop_w+1)/2), (0, 0)))

        # Random horizontal flip
        if self._distort:
            if random.choice(range(2)) == 1:
                for i in range(image.shape[2]):
                    image[:,:,i] = np.fliplr(image[:,:,i])

        # Image whitening
        mean = np.mean(image, axis=(0,1), dtype=np.float32)
        std = np.std(image, axis=(0,1), dtype=np.float32)
        output_image = (image - mean) / std

        return output_image

    def _preprocess_images(self, input_images):
        output_images = np.zeros_like(input_images, dtype=np.float32)
        for i in range(output_images.shape[0]):
            output_images[i] = self._preprocess_image(input_images[i])

        return output_images

    def get_inputs(self, batch_size):
        """
        Return's tensors containing a batch of images and labels
        """
        images_batch, labels_batch = self.queue.dequeue_many(batch_size)
        if not CIFAR100Runner._image_summary_added:
            tf.summary.image('images', images_batch)
            CIFAR100Runner._image_summary_added = True

        return images_batch, labels_batch

    def data_iterator(self):
        idxs_idx = 0
        idxs = np.arange(0, self.size)
        if self._shuffle:
            random.shuffle(idxs)

        while True:
            images_batch = []
            labels_batch = []
            batch_cnt = 0
            while True:
                if idxs_idx + (self.image_per_thread - batch_cnt) < self.size:
                    temp_cnt = self.image_per_thread - batch_cnt
                else:
                    temp_cnt = self.size - idxs_idx

                images_batch.extend(self._images[idxs[idxs_idx:idxs_idx+temp_cnt]])
                labels_batch.extend(self._labels[idxs[idxs_idx:idxs_idx+temp_cnt]])
                idxs_idx += temp_cnt
                batch_cnt += temp_cnt

                if idxs_idx == self.size:
                    idxs_idx = 0
                    if self._shuffle:
                        random.shuffle(idxs)

                if batch_cnt == self.image_per_thread:
                    break
            yield images_batch, labels_batch

    def thread_main(self, sess, iterator):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        while True:
            images_val, labels_val = iterator.next()
            process_images_val = self._preprocess_images(images_val)
            sess.run(self.enqueue_op, feed_dict={self.dataX:process_images_val, self.dataY:labels_val})

    def start_threads(self, sess, n_threads=1):
        """ Start background threads to feed queue """
        iterator = ThreadsafeIter(self.data_iterator())
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,iterator,))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads
