# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Frechet Inception Distance (FID)."""

import os
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc

#----------------------------------------------------------------------------

class FID(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, split='test', num_real_images=None, mirror_augment=False, num_repeats=1, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.split = split
        self.num_real_images = num_real_images
        self.num_repeats = num_repeats
        self.mirror_augment = mirror_augment
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, Gs_kwargs, num_gpus):
        if self.num_real_images is None:
            self.num_real_images = self._get_dataset_obj(split=self.split).num_samples
        if self.num_images is None:
            self.num_images = self.num_real_images
        num_channels = Gs.output_shape[1]
        minibatch_size = num_gpus * self.minibatch_per_gpu
        inception = misc.load_pkl('http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/inception_v3_features.pkl')

        # Calculate statistics for reals.
        cache_file = self._get_cache_file_for_reals(prefix='fid', num_images=self.num_real_images, split=self.split, mirror_augment=self.mirror_augment)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        if os.path.isfile(cache_file):
            mu_real, sigma_real = misc.load_pkl(cache_file)
        else:
            activations = np.empty([self.num_real_images, inception.output_shape[1]], dtype=np.float32)
            for idx, images in enumerate(self._iterate_reals(minibatch_size=minibatch_size, split=self.split, mirror_augment=self.mirror_augment)):
                begin = idx * minibatch_size
                end = min(begin + minibatch_size, self.num_real_images)
                images = images[:end-begin]
                if num_channels == 1:
                    images = np.repeat(images, 3, axis=1)
                activations[begin:end] = inception.run(images, num_gpus=num_gpus if images.shape[0] % num_gpus == 0 else 1, assume_frozen=True)
                if end == self.num_real_images:
                    break
            mu_real = np.mean(activations, axis=0)
            sigma_real = np.cov(activations, rowvar=False)
            misc.save_pkl((mu_real, sigma_real), cache_file)

        # Construct TensorFlow graph.
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                inception_clone = inception.clone()
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                labels = self._get_random_labels_tf(self.minibatch_per_gpu, split=self.split)
                images = Gs_clone.get_output_for(latents, labels, **Gs_kwargs)
                if num_channels == 1:
                    images = tf.repeat(images, 3, axis=1)
                images = tflib.convert_images_to_uint8(images)
                result_expr.append(inception_clone.get_output_for(images))

        activations = np.empty([self.num_images, inception.output_shape[1]], dtype=np.float32)
        results = []
        for _ in range(self.num_repeats):
            # Calculate statistics for fakes.
            for begin in range(0, self.num_images, minibatch_size):
                self._report_progress(begin, self.num_images)
                end = min(begin + minibatch_size, self.num_images)
                activations[begin:end] = np.concatenate(tflib.run(result_expr), axis=0)[:end-begin]
            mu_fake = np.mean(activations, axis=0)
            sigma_fake = np.cov(activations, rowvar=False)

            # Calculate FID.
            m = np.square(mu_fake - mu_real).sum()
            s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
            dist = m + np.trace(sigma_fake + sigma_real - 2*s)
            results.append(np.real(dist))
        self._report_result(np.mean(results))
        if self.num_repeats > 1:
            self._report_result(np.std(results), suffix='-std')

#----------------------------------------------------------------------------
