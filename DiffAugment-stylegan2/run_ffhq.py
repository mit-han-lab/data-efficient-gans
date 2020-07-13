# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import copy
import os
import sys

import dnnlib
from dnnlib import tflib
from dnnlib import EasyDict

from metrics import metric_base
from metrics.metric_defaults import metric_defaults
from training import dataset_tool
from training import misc

# ----------------------------------------------------------------------------


def run(dataset, resolution, result_dir, DiffAugment, num_gpus, batch_size, total_kimg, ema_kimg, num_samples, gamma, fmap_base, fmap_max, latent_size, mirror_augment, impl, metrics, resume, resume_kimg, num_repeats, eval):
    train = EasyDict(run_func_name='training.training_loop.training_loop')  # Options for training loop.
    G = EasyDict(func_name='training.networks_stylegan2.G_main')       # Options for generator network.
    D = EasyDict(func_name='training.networks_stylegan2.D_stylegan2')  # Options for discriminator network.
    G_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for generator optimizer.
    D_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for discriminator optimizer.
    loss_args = EasyDict(func_name='training.loss.ns_r1_DiffAugment')          # Options for loss.
    sched = EasyDict()                                                     # Options for TrainingSchedule.
    grid = EasyDict(size='4k', layout='random')                           # Options for setup_snapshot_image_grid().
    sc = dnnlib.SubmitConfig()                                          # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': 1000}                                   # Options for tflib.init_tf().

    # preprocess dataset into tfrecords if necessary
    dataset = dataset_tool.create_dataset(dataset, resolution)

    train.total_kimg = total_kimg
    train.mirror_augment = mirror_augment
    metrics = [metric_defaults[x] for x in metrics]
    metric_args = EasyDict(cache_dir=dataset, num_repeats=num_repeats)

    desc = 'DiffAugment-stylegan2' if DiffAugment else 'stylegan2'
    dataset_args = EasyDict(tfrecord_dir=dataset, resolution=resolution, from_tfrecords=True)
    desc += '-' + os.path.basename(dataset)
    if resolution is not None:
        desc += '-{}'.format(resolution)

    if num_samples is not None:
        dataset_args.num_samples = num_samples
        desc += '-{}samples'.format(num_samples)

    if batch_size is not None:
        desc += '-batch{}'.format(batch_size)
    else:
        batch_size = 32
    assert batch_size % num_gpus == 0
    sc.num_gpus = num_gpus
    desc += '-%dgpu' % num_gpus
    sched.minibatch_size_base = batch_size
    sched.minibatch_gpu_base = batch_size // num_gpus

    G.impl = D.impl = impl
    if fmap_base is not None:
        G.fmap_base = D.fmap_base = fmap_base
        desc += '-fmap{}'.format(fmap_base)
    if fmap_max is not None:
        G.fmap_max = D.fmap_max = fmap_max
        desc += '-fmax{}'.format(fmap_max)
    if latent_size is not None:
        G.latent_size = G.mapping_fmaps = G.dlatent_size = latent_size
        desc += '-latent{}'.format(latent_size)

    if gamma is not None:
        loss_args.gamma = gamma
        desc += '-gamma{}'.format(gamma)
    if DiffAugment:
        loss_args.policy = DiffAugment
        desc += '-' + DiffAugment.replace(',', '-')

    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    kwargs = EasyDict(train)
    kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, loss_args=loss_args)
    kwargs.update(dataset_args=dataset_args, sched_args=sched, grid_args=grid, metric_arg_list=metrics, tf_config=tf_config)
    kwargs.update(resume_pkl=resume, resume_kimg=resume_kimg, resume_with_new_nets=True)
    kwargs.update(metric_args=metric_args)
    if ema_kimg is not None:
        kwargs.update(G_ema_kimg=ema_kimg)
    kwargs.submit_config = copy.deepcopy(sc)
    kwargs.submit_config.run_dir_root = result_dir
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)

# ----------------------------------------------------------------------------


def run_eval(dataset, resolution, result_dir, DiffAugment, num_gpus, batch_size, total_kimg, ema_kimg, num_samples, gamma, fmap_base, fmap_max, latent_size, mirror_augment, impl, metrics, resume, resume_kimg, num_repeats, eval):
    dataset = dataset_tool.create_dataset(dataset, resolution)
    print('Evaluating metrics "%s" for "%s"...' % (','.join(metrics), resume))
    tflib.init_tf()
    dataset_args = dnnlib.EasyDict(tfrecord_dir=dataset, shuffle_mb=0, from_tfrecords=True)
    metric_group = metric_base.MetricGroup([metric_defaults[metric] for metric in metrics], num_repeats=num_repeats)
    metric_group.run(resume, dataset_args=dataset_args, num_gpus=num_gpus)

# ----------------------------------------------------------------------------


def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _parse_comma_sep(s):
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='Train StyleGAN2 + DiffAugment.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', help='Training dataset directory', required=True)
    parser.add_argument('--resolution', help='Specifies resolution', default=None, type=int)
    parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser.add_argument('--DiffAugment', help='Comma-separated list of DiffAugment policy', default='color,translation,cutout')
    parser.add_argument('--num-gpus', help='Number of GPUs (default: %(default)s)', default=1, type=int, metavar='N')
    parser.add_argument('--batch-size', help='Batch size', default=None, type=int, metavar='N')
    parser.add_argument('--total-kimg', help='Training length in thousands of images (default: %(default)s)', metavar='KIMG', default=25000, type=int)
    parser.add_argument('--ema-kimg', help='Half-life of exponential moving average in thousands of images', metavar='KIMG', default=None, type=int)
    parser.add_argument('--num-samples', help='Number of samples', default=None, type=int)
    parser.add_argument('--gamma', help='R1 regularization weight', default=1, type=float)
    parser.add_argument('--fmap-base', help='Number of feature maps', default=8192, type=int)
    parser.add_argument('--fmap-max', help='Maximum number of feature maps', default=None, type=int)
    parser.add_argument('--latent-size', help='Latent size', default=None, type=int)
    parser.add_argument('--mirror-augment', help='Mirror augment (default: %(default)s)', default=True, metavar='BOOL', type=_str_to_bool)
    parser.add_argument('--impl', help='Custom op implementation (default: %(default)s)', default='cuda')
    parser.add_argument('--metrics', help='Comma-separated list of metrics or "none" (default: %(default)s)', default='fid50k-train', type=_parse_comma_sep)
    parser.add_argument('--resume', help='Resume checkpoint path', default=None)
    parser.add_argument('--resume-kimg', help='Resume training length', default=0, type=int)
    parser.add_argument('--num-repeats', help='Repeats of evaluation runs (default: %(default)s)', default=1, type=int, metavar='N')
    parser.add_argument('--eval', help='Evalulate mode?', action='store_true')

    args = parser.parse_args()

    for metric in args.metrics:
        if metric not in metric_defaults:
            print('Error: unknown metric \'%s\'' % metric)
            sys.exit(1)

    if args.eval:
        run_eval(**vars(args))
    else:
        run(**vars(args))

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
