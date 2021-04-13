# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import copy
import torch
import dnnlib
from . import metric_utils
from tqdm.auto import tqdm
from DiffAugment_pytorch import DiffAugment


def compute_accuracy(opts, batch_size=32):
    D = copy.deepcopy(opts.D).eval().requires_grad_(False).to(opts.device)
    train_dataset = opts.train_dataset
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)

    train_correct = 0
    train_all = 0

    for i, (train_img, train_c) in enumerate(tqdm(train_dataloader)):
        train_img = train_img.to(opts.device).to(torch.float32) / 127.5 - 1
        train_c = train_c.to(opts.device)
        gen_logits = D(train_img, train_c)
        train_all += train_img.shape[0]
        train_correct += torch.sum(gen_logits > 0).detach().item()
    
    train_accuracy = train_correct / train_all

    if opts.validation_dataset_kwargs != {}:
        validation_dataset = opts.validation_dataset
        validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size)

        validation_correct = 0
        validation_all = 0

        for i, (validation_img, validation_c) in enumerate(tqdm(validation_dataloader)):
            validation_img = validation_img.to(opts.device).to(torch.float32) / 127.5 - 1
            validation_c = validation_c.to(opts.device)
            gen_logits = D(validation_img, validation_c)
            validation_all += validation_img.shape[0]
            validation_correct += torch.sum(gen_logits > 0).detach().item()
        validation_accuracy = validation_correct / validation_all
    else:
        validation_accuracy = None

    return train_accuracy, validation_accuracy


def compute_accuracy_generated(opts, batch_size=32, diff_aug=False):
    D = copy.deepcopy(opts.D).eval().requires_grad_(False).to(opts.device)
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)

    train_correct = 0
    train_all = 0
    if opts.validation_dataset_kwargs != {}:
        all_z = torch.randn([len(opts.validation_dataset), G.z_dim], device=opts.device)
    else:
        all_z = torch.randn([10000, G.z_dim], device=opts.device)
    z_loader = torch.utils.data.DataLoader(dataset=all_z, batch_size=batch_size)
    for i, z in enumerate(tqdm(z_loader)):
        fake_img = G(z, torch.empty([batch_size, 0], device=opts.device))
        if diff_aug:
            fake_img = DiffAugment(fake_img, policy=opts.loss_kwargs.diffaugment)
        logits = D(fake_img, torch.empty([batch_size, 0], device=opts.device))
        train_all += fake_img.shape[0]
        train_correct += torch.sum(logits <= 0).detach().item()
    result = train_correct / train_all
    return result
