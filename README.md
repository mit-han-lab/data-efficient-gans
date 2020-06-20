# Data-Efficient GANs with DiffAugment

**[NOTE]** Our DiffAugment-biggan repo is coming!

This repository contains our implementation of Differentiable Augmentation (DiffAugment) in both PyTorch and Tensorflow. It can be used to significantly improve the data efficiency for GAN training. We have provided the training code of [DiffAugment-stylegan2](https://github.com/mit-han-lab/data-efficient-gans/tree/master/DiffAugment-stylegan2) (Tensorflow).

![few_shot-comparison](imgs/few_shot-comparison.jpg)

*Few-shot generation without pre-training. With DiffAugment, our model can generate high-fidelity images using only 100 Obama portraits, grumpy cats, or pandas from our collected 100-shot datasets, 160 cats or 389 dogs from the AnimalFace dataset at 256×256 resolution.*

![cifar10-results](imgs/cifar10-results.jpg)

*Unconditional generation results on CIFAR-10. StyleGAN2’s performance drastically degrades given less training data. With DiffAugment, we are able to roughly match its FID and outperform its Inception Score (IS) using only **20%** training data.*

Differentiable Augmentation for Data-Efficient GAN Training<br>
[Shengyu Zhao](https://scholar.google.com/citations?user=gLCdw70AAAAJ), [Zhijian Liu](http://zhijianliu.com/), [Ji Lin](http://linji.me/), [Jun-Yan Zhu](https://people.csail.mit.edu/junyanz/), and [Song Han](https://songhan.mit.edu/)<br>
MIT, Tsinghua University, Adobe Research<br>
[arXiv](https://arxiv.org/pdf/2006.10738.pdf)


## Overview

![method](imgs/method.jpg)

*Overview of DiffAugment for updating D (left) and G (right). DiffAugment applies the augmentation T to both the real sample x and the generated output G(z). When we update G, gradients need to be back-propagated through T, which requires T to be differentiable w.r.t. the input.*

## DiffAugment for StyleGAN2

To run *StyleGAN2 + DiffAugment* for unconditional generation on CIFAR and few-shot generation, please refer to the [DiffAugment-stylegan2](https://github.com/mit-han-lab/data-efficient-gans/tree/master/DiffAugment-stylegan2) README.

## Using DiffAugment for Your Own Training

To help you use DiffAugment in your own codebase, we provide portable DiffAugment operations of both TensorFlow and PyTorch versions in [DiffAugment_tf.py](https://github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment_tf.py) and [DiffAugment_pytorch.py](https://github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment_pytorch.py). Generally, DiffAugment can be easily adopted in any model by substituting every *D(x)* with *D(T(x))*, where *x* can be real images or fake images, *D* is the discriminator, and *T* is the DiffAugment operation. For example,

```
from DiffAugment_pytorch import DiffAugment
# from DiffAugment_tf import DiffAugment
policy = 'color,translation,cutout' # If your dataset is as small as ours (e.g.,
# 100 images), we recommend using the strongest DiffAugment:  Color + Translation + Cutout.
# For large datasets, try using a subset of transformations in ['color', 'translation', 'cutout'].
# Welcome to discover more DiffAugment transformations!

...
# Training loop
reals = sample_real_images() # a batch of real images
fakes = generate_fake_images() # a batch of fake images
real_scores = Discriminator(DiffAugment(reals, policy=policy))
fake_scores = Discriminator(DiffAugment(fakes, policy=policy))
# Calculating loss based on real_scores and fake_scores...
...
```

## Citation
If you use this code in your research, please cite our paper:
```
@article{zhao2020diffaugment,
  title={Differentiable Augmentation for Data-Efficient GAN Training},
  author={Zhao, Shengyu and Liu, Zhijian and Lin, Ji and Zhu, Jun-Yan and Han, Song},
  journal={arXiv preprint arXiv:2006.10738},
  year={2020}
}
```
