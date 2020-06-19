# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Tool for creating multi-resolution TFRecords datasets."""

# pylint: disable=too-many-lines
import os
import zipfile
import glob
import numpy as np
import tensorflow as tf
import PIL.Image
import dnnlib

# ----------------------------------------------------------------------------


def error(msg):
    print('Error: ' + msg)
    exit(1)

# ----------------------------------------------------------------------------


class TFRecordExporter:
    def __init__(self, tfrecord_dir, expected_images, print_progress=True, progress_interval=10):
        self.tfrecord_dir = tfrecord_dir
        self.tfr_prefix = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        self.expected_images = expected_images
        self.cur_images = 0
        self.shape = None
        self.resolution_log2 = None
        self.tfr_writers = []
        self.print_progress = print_progress
        self.progress_interval = progress_interval

        if self.print_progress:
            print('Creating dataset "%s"' % tfrecord_dir)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert os.path.isdir(self.tfrecord_dir)

    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    def choose_shuffled_order(self):  # Note: Images and labels must be added in shuffled order.
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order

    def set_shape(self, shape):
        self.shape = shape
        self.resolution_log2 = int(np.log2(self.shape[1]))
        # assert self.shape[0] in [1, 3]
        assert self.shape[1] == self.shape[2]
        assert self.shape[1] == 2**self.resolution_log2
        tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
        tfr_file = self.tfr_prefix + '.tfrecords'
        self.tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))

    def add_image(self, img):
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.set_shape(img.shape)
        assert img.shape == self.shape
        for lod, tfr_writer in enumerate(self.tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def add_labels(self, labels):
        if self.print_progress:
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        assert labels.shape[0] == self.cur_images
        with open(self.tfr_prefix + '-rxx.labels', 'wb') as f:
            np.save(f, labels.astype(np.int32))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

# ----------------------------------------------------------------------------


def create_from_images(dataset, resolution, tfrecord_dir=None, shuffle=True):
    if dataset in ['100-shot-obama', '100-shot-grumpy_cat', '100-shot-panda']:
        image_dir = 'datasets/{}'.format(dataset)
        if not glob.glob(os.path.join(image_dir, '*.tfrecords')):
            try:
                os.makedirs(image_dir)
            except:
                pass
            dataset_url = 'https://hanlab.mit.edu/projects/data-efficient-gans/datasets/{}.zip'.format(dataset)
            zip_path = dnnlib.util.open_url(dataset_url, cache_dir='.stylegan2-cache', return_path=True)
            with zipfile.ZipFile(zip_path, 'r') as f:
                f.extractall(image_dir)
    else:
        image_dir = dataset
    assert os.path.isdir(image_dir)
    if tfrecord_dir is None:
        tfrecord_dir = image_dir

    print('Loading images from "%s"' % image_dir)
    image_filenames = sorted(glob.glob(os.path.join(image_dir, '*')))
    image_filenames = [fname for fname in image_filenames if fname.split('.')[-1] not in ['pkl', 'tfrecords']]
    if len(image_filenames) == 0:
        error('No input images found')

    img = np.asarray(PIL.Image.open(image_filenames[0]))
    if resolution is None:
        resolution = img.shape[0]
        if img.shape[1] != resolution:
            error('Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        error('Input image resolution must be a power-of-two')
    channels = img.shape[2] if img.ndim == 3 else 1
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')

    with TFRecordExporter(tfrecord_dir, len(image_filenames)) as tfr:
        order = tfr.choose_shuffled_order() if shuffle else np.arange(len(image_filenames))
        for idx in range(order.size):
            img = PIL.Image.open(image_filenames[order[idx]])
            if resolution is not None:
                img = img.resize((resolution, resolution), PIL.Image.ANTIALIAS)
            img = np.asarray(img)
            if channels == 1 or len(img.shape) == 2:
                img = np.stack([img] * channels)  # HW => CHW
            else:
                img = img.transpose([2, 0, 1])  # HWC => CHW
            tfr.add_image(img)
    return tfrecord_dir, len(image_filenames)

# ----------------------------------------------------------------------------
