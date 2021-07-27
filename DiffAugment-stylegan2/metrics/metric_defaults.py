# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Default metric definitions."""

from dnnlib import EasyDict

#----------------------------------------------------------------------------

metric_defaults = EasyDict([(args.name, args) for args in [
    EasyDict(name='fid10k',       func_name='metrics.frechet_inception_distance.FID', num_images=10000, minibatch_per_gpu=8),
    EasyDict(name='fid5k-train',  func_name='metrics.frechet_inception_distance.FID', num_images=5000,  split='train', minibatch_per_gpu=8),
    EasyDict(name='fid50k-train', func_name='metrics.frechet_inception_distance.FID', num_images=50000, split='train', minibatch_per_gpu=8),
    EasyDict(name='is10k',        func_name='metrics.inception_score.IS',             num_images=10000, split='train', num_splits=10, minibatch_per_gpu=8),
]])

#----------------------------------------------------------------------------
