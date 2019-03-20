import numpy as np
import cv2
import os
import inspect
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from skimage import io
from skimage import transform

curr_filename = inspect.getfile(inspect.currentframe())
root_dir = os.path.dirname(os.path.abspath(curr_filename))
task3_img = 'ISIC2018_Task3_Training_Input_Original'
task3_gt = 'ISIC2018_Task3_Training_GroundTruth'
task3_gt_fname = 'ISIC2018_Task3_Training_GroundTruth.csv'
task3_sup_fname = 'ISIC2018_Task3_Training_LesionGroupings.csv'
data_dir = os.path.join(root_dir, 'data')
task3_img_dir = os.path.join(data_dir, task3_img)
task3_gt_dir = os.path.join(data_dir, task3_gt)
cached_data_dir = os.path.join(root_dir, 'data', 'cache')
task3_images_npy_prefix = 'task3_images'

task3_image_ids = list()
if os.path.isdir(task3_img_dir):
    task3_image_ids = [fname.rsplit('.', maxsplit=1)[0] for fname in os.listdir(task3_img_dir)
                       if fname.startswith('ISIC') and fname.lower().endswith('.jpg')]

    task3_image_ids.sort()

print(task3_gt_dir)

def load_image_by_id(image_id, fname_fn, from_dir, output_size=None, return_size=False):
    img_fnames = fname_fn(image_id)
    if isinstance(img_fnames, str):
        img_fnames = [img_fnames, ]

    assert isinstance(img_fnames, tuple) or isinstance(img_fnames, list)

    images = []
    image_sizes = []

    for img_fname in img_fnames:
        img_fname = os.path.join(from_dir, img_fname)
        if not os.path.exists(img_fname):
            raise FileNotFoundError('img %s not found' % img_fname)
        image = io.imread(img_fname)

        image_sizes.append(np.asarray(image.shape[:2]))

        if output_size:
            image = transform.resize(image, (output_size, output_size),
                                     order=1, mode='constant',
                                     cval=0, clip=True,
                                     preserve_range=True,
                                     anti_aliasing=True)
        image = image.astype(np.uint8)
        images.append(image)

    if return_size:
        if len(images) == 1:
            return images[0], image_sizes[0]
        else:
            return np.stack(images, axis=-1), image_sizes

    if len(images) == 1:
        return images[0]
    else:
        return np.stack(images, axis=-1)  # masks


def load_images(image_ids, from_dir, output_size=None, fname_fn=None, verbose=True, return_size=False):
    images = []

    if verbose:
        print('loading images from', from_dir)

    if return_size:

        image_sizes = []
        for image_id in tqdm(image_ids):
            image, image_size = load_image_by_id(image_id,
                                                 from_dir=from_dir,
                                                 output_size=output_size,
                                                 fname_fn=fname_fn,
                                                 return_size=True)
            images.append(image)
            image_sizes.append(image_size)

        return images, image_sizes


    else:
        for image_id in tqdm(image_ids):
            image = load_image_by_id(image_id,
                                     from_dir=from_dir,
                                     output_size=output_size,
                                     fname_fn=fname_fn)
            images.append(image)

        return images

def load_task3_training_images(output_size=None):
    suffix = '' if output_size is None else '_%d' % output_size
    images_npy_filename = os.path.join(cached_data_dir, '%s%s.npy' % (task3_images_npy_prefix, suffix))

    if os.path.exists(images_npy_filename):
        images = np.load(images_npy_filename)
    else:
        images = load_images(image_ids=task3_image_ids,
                             from_dir=task3_img_dir,
                             output_size=output_size,
                             fname_fn=lambda x: '%s.jpg' % x)
        images = np.stack(images).astype(np.uint8)
        np.save(images_npy_filename, images)
    return images

def load_task3_training_labels():
    # image, MEL, NV, BCC, AKIEC, BKL, DF, VASC
    labels = []
    with open(os.path.join(task3_gt_dir, task3_gt_fname), 'r') as f:
        for i, line in tqdm(enumerate(f.readlines()[1:])):
            fields = line.strip().split(',')
            labels.append([eval(field) for field in fields[1:]])
        labels = np.stack(labels, axis=0)
    return labels

def partition_data(x, y, k=5, i=0, test_split=1. / 6, seed=42):
    assert isinstance(k, int) and isinstance(i, int) and 0 <= i < k

    n = x.shape[0]

    n_set = int(n * (1. - test_split)) // k
    # divide the data into (k + 1) sets, -1 is test set, [0, k) are for train and validation
    indices = np.array([i for i in range(k) for _ in range(n_set)] +
                       [-1] * (n - n_set * k),
                       dtype=np.int8)

    np.random.seed(seed)
    np.random.shuffle(indices)

    valid_indices = (indices == i)
    test_indices = (indices == -1)
    train_indices = ~(valid_indices | test_indices)

    x_valid = x[valid_indices]
    y_valid = y[valid_indices]

    x_train = x[train_indices]
    y_train = y[train_indices]

    x_test = x[test_indices]
    y_test = y[test_indices]

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

def partition_task3_data(x, y, k=1, i=0, test_split=1. / 6, seed=42):
    assert isinstance(k, int) and isinstance(i, int) and 0 <= i < k

    fname = os.path.join(task3_gt_dir, task3_sup_fname)
    assert os.path.exists(fname)

    df = pd.read_csv(os.path.join(task3_gt_dir, task3_sup_fname))
    grouped = df.groupby('lesion_id', sort=True)
    lesion_ids = []
    for name, group in grouped:
        image_ids = group.image.tolist()
        lesion_ids.append([name, image_ids])

    # shuffle lesion ids
    np.random.seed(seed)
    n = len(lesion_ids)
    indices = np.random.permutation(n)

    image_ids = [image_id for idx in indices for image_id in lesion_ids[idx][1]]
    n = len(image_ids)
    n_set = int(n * (1. - test_split)) // k
    # divide the data into (k + 1) sets, -1 is test set, [0, k) are for train and validation
    indices = [i for i in range(k) for _ in range(n_set)] + [-1] * (n - n_set * k)

    indices = list(zip(indices, image_ids))
    indices.sort(key=lambda x: x[1])
    indices = np.array([idx for idx, image_id in indices], dtype=np.uint)

    valid_indices = (indices == i)
    test_indices = (indices == -1)
    train_indices = ~(valid_indices | test_indices)

    x_valid = x[valid_indices]
    y_valid = y[valid_indices]

    x_train = x[train_indices]
    y_train = y[train_indices]

    x_test = x[test_indices]
    y_test = y[test_indices]

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

def load_training_data(task_idx,
                       output_size=None,
                       num_partitions=5,
                       idx_partition=0,
                       test_split=0.):
    assert isinstance(task_idx, int) and 0 < task_idx <= 3
    if task_idx == 1:
        x = load_task12_training_images(output_size=output_size)
        y = load_task1_training_masks(output_size=output_size)
        return partition_data(x=x, y=y, k=num_partitions, i=idx_partition, test_split=test_split)

    elif task_idx == 2:
        x = load_task12_training_images(output_size=output_size)
        y = load_task2_training_masks(output_size=output_size)
        return partition_data(x=x, y=y, k=num_partitions, i=idx_partition, test_split=test_split)

    else:
        x = load_task3_training_images(output_size=output_size)
        y = load_task3_training_labels()
        return partition_task3_data(x=x, y=y, k=num_partitions, i=idx_partition, test_split=test_split)