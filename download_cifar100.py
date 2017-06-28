import os
import sys
import random
import tarfile
import cPickle as pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", help="The directory to save splited CIFAR-100 dataset")
args = parser.parse_args()

# CIFAR-100 download parameters
cifar_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
cifar_dpath = 'cifar100'
cifar_py_name = 'cifar-100-python'
cifar_fname = cifar_py_name + '.tar.gz'

# CIFAR-100 dataset train/val split parameters
dataset_path = 'cifar100/train_val_split' if not args.dataset_path else args.dataset_path
dataset_path = os.path.expanduser(dataset_path)
num_train_instance = 45000
num_val_instance = 50000 - num_train_instance
num_test_instance = 10000

def download_file(url, path):
    import urllib2
    file_name = url.split('/')[-1]
    u = urllib2.urlopen(url)
    f = open(os.path.join(path, file_name), 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (file_name, file_size)

    download_size = 0
    block_size = 8192
    while True:
        buf = u.read(block_size)
        if not buf:
            break
        download_size += len(buf)
        f.write(buf)
        status = "\r%12d  [%3.2f%%]" % (download_size, download_size * 100. / file_size)
        print status,
        sys.stdout.flush()
    f.close()

# Check if the dataset split already exists
if os.path.exists(dataset_path) and os.path.exists(os.path.join(dataset_path, 'train')):
    print('CIFAR-100 train/val split exists\nNothing to be done... Quit!')
    sys.exit(0)

# Download and extract CIFAR-100
if not os.path.exists(os.path.join(cifar_dpath, cifar_py_name)) \
        or not os.path.exists(os.path.join(cifar_dpath, cifar_py_name, 'train')):
    print('Downloading CIFAR-100')
    if not os.path.exists(cifar_dpath):
        os.makedirs(cifar_dpath)
    tar_fpath = os.path.join(cifar_dpath, cifar_fname)
    if not os.path.exists(tar_fpath) or os.path.getsize(tar_fpath) != 169001437:
        download_file(cifar_url, cifar_dpath)
    print('Extracting CIFAR-100')
    with tarfile.open(tar_fpath) as tar:
        tar.extractall(path=cifar_dpath)

# Load the dataset and split
print('Load CIFAR-100 dataset')
with open(os.path.join(cifar_dpath, cifar_py_name, 'train')) as fd:
    train_orig = pickle.load(fd)
    train_orig_data = train_orig['data']
    train_orig_label = np.array(train_orig['fine_labels'], dtype=np.uint8)
with open(os.path.join(cifar_dpath, cifar_py_name, 'test')) as fd:
    test_orig = pickle.load(fd)
    test_orig_data = test_orig['data']
    test_orig_label = np.array(test_orig['fine_labels'], dtype=np.uint8)

# Split the dataset
print('Split the dataset')
train_val_idxs = range(50000)
random.shuffle(train_val_idxs)
train = {'data': train_orig_data[train_val_idxs[:num_train_instance], :],
         'labels': train_orig_label[train_val_idxs[:num_train_instance]]}
val = {'data': train_orig_data[train_val_idxs[num_train_instance:] ,:],
       'labels': train_orig_label[train_val_idxs[num_train_instance:]]}
train_val = {'data':train_orig_data, 'labels':train_orig_label}
test = {'data':test_orig_data, 'labels':test_orig_label}

# Save the dataset split
print('Save the dataset split')
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
for name, data in zip(['train', 'val', 'train_val', 'test'], [train, val, train_val, test]):
    print('[%s] ' % name + ', '.join(['%s: %s' % (k, str(v.shape)) for k, v in data.iteritems()]))
    with open(os.path.join(dataset_path, name), 'wb') as fd:
        pickle.dump(data, fd)

print 'done'
