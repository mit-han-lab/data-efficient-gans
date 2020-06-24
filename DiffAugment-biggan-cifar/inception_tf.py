import fnmatch
import importlib
import inspect
import scipy
import numpy as np
import os
import shutil
import sys
import types
import io
import pickle
import re
import requests
import html
import hashlib
import glob
import uuid
from typing import Any, List, Tuple, Union
import torch

import dnnlib
import dnnlib.tflib
import utils


def prepare_inception_metrics(dataset, parallel, config):
    dataset = dataset.strip('_hdf5')
    dnnlib.tflib.init_tf()
    inception_v3_features = dnnlib.util.load_pkl(
        'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/inception_v3_features.pkl')
    inception_v3_softmax = dnnlib.util.load_pkl(
        'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/inception_v3_softmax.pkl')
    try:
        mu_real, sigma_real = dnnlib.util.load_pkl(dataset + '_inception_moments.pkl')
    except:
        print('Calculating inception features for the training set...')
        loader = utils.get_data_loaders(
            **{**config, 'train': False, 'mirror_augment': False,
            'use_multiepoch_sampler': False, 'load_in_mem': False, 'pin_memory': False})[0]
        pool = []
        num_gpus = torch.cuda.device_count()
        for images, _ in loader:
            images = ((images.numpy() * 0.5 + 0.5)
                      * 255 + 0.5).astype(np.uint8)
            pool.append(inception_v3_features.run(images,
                                                  num_gpus=num_gpus, assume_frozen=True))
        pool = np.concatenate(pool)
        mu_real, sigma_real = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
        dnnlib.util.save_pkl((mu_real, sigma_real), dataset + '_inception_moments.pkl')
        mu_real, sigma_real = dnnlib.util.load_pkl(dataset + '_inception_moments.pkl')

    def get_inception_metrics(sample, num_inception_images, num_splits=10, prints=True, use_torch=True):
        pool, logits = accumulate_inception_activations(
            sample, inception_v3_features, inception_v3_softmax, num_inception_images)
        IS_mean, IS_std = calculate_inception_score(logits, num_splits)
        mu_fake, sigma_fake = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(
            np.dot(sigma_fake, sigma_real), disp=False)  # pylint: disable=no-member
        dist = m + np.trace(sigma_fake + sigma_real - 2*s)
        FID = np.real(dist)
        return IS_mean, IS_std, FID
    return get_inception_metrics


def accumulate_inception_activations(sample, inception_v3_features, inception_v3_softmax, num_inception_images):
    pool, logits = [], []
    cnt = 0
    num_gpus = torch.cuda.device_count()
    while cnt < num_inception_images:
        images, _ = sample()
        images = ((images.cpu().numpy() * 0.5 + 0.5)
                  * 255 + 0.5).astype(np.uint8)
        pool.append(inception_v3_features.run(images,
                                              num_gpus=num_gpus, assume_frozen=True))
        logits.append(inception_v3_softmax.run(images,
                                               num_gpus=num_gpus, assume_frozen=True))
        cnt += images.shape[0]
    return np.concatenate(pool), np.concatenate(logits, 0)


def calculate_inception_score(pred, num_splits=10):
    scores = []
    for index in range(num_splits):
        pred_chunk = pred[index * (pred.shape[0] // num_splits)                          : (index + 1) * (pred.shape[0] // num_splits), :]
        kl_inception = pred_chunk * \
            (np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
        kl_inception = np.mean(np.sum(kl_inception, 1))
        scores.append(np.exp(kl_inception))
    return np.mean(scores), np.std(scores)
