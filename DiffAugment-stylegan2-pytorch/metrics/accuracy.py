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

def compute_accuracy(opts, batch_size=32):
    D = copy.deepcopy(opts.D).eval().requires_grad_(False).to(opts.device)
    train_dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)

    train_correct = 0
    train_all = 0

    for i, (train_img, train_c) in enumerate(tqdm(train_dataloader)):
        train_img = train_img.to(opts.device)
        train_c = train_c.to(opts.device)
        gen_logits = D(train_img, train_c)
        train_all += train_img.shape[0]
        train_correct += torch.sum(gen_logits > 0).detach().item()
    
    train_accuracy = train_correct / train_all

    if opts.validation_dataset_kwargs != {}:
        validation_dataset = dnnlib.util.construct_class_by_name(**opts.validation_dataset_kwargs)
        validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size)

        validation_correct = 0
        validation_all = 0

        for i, (validation_img, validation_c) in enumerate(tqdm(validation_dataloader)):
            validation_img = validation_img.to(opts.device)
            validation_c = validation_c.to(opts.device)
            gen_logits = D(validation_img, validation_c)
            validation_all += validation_img.shape[0]
            validation_correct += torch.sum(gen_logits > 0).detach().item()
        validation_accuracy = validation_correct / validation_all
    else:
        validation_accuracy = None
    

    return train_accuracy, validation_accuracy
