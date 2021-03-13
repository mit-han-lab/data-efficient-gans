# DiffAugment for StyleGAN2 (PyTorch)

This repo is implemented upon [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) with minimal modifications to train and load DiffAugment-stylegan2 models in PyTorch. Please check the [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) README for the dependencies and the other usages of this codebase.

## Low-Shot Generation

The following command is an example of training StyleGAN2 with the default *Color + Translation + Cutout* DiffAugment on 100-shot Obama with 1 GPU. See [here](https://hanlab.mit.edu/projects/data-efficient-gans/datasets/) for a list of our provided low-shot datasets. You may also prepare your own dataset and specify the path to your image folder.
```bash
python train.py --outdir=training-runs --data=https://hanlab.mit.edu/projects/data-efficient-gans/datasets/100-shot-obama.zip --gpus=1
```

<img src="../imgs/low-shot-interp.jpg" width="1000px"/>

## Pre-Trained Models

The following command is an example of generation with our pre-trained 100-shot Obama model. It will automatically convert a TensorFlow model to the PyTorch version. See [here](https://hanlab.mit.edu/projects/data-efficient-gans/models/) for a list of our provided pre-trained models. You may also use `legacy.py` to manually convert our pre-trained models to the compatible PyTorch version.
```bash
python generate.py --outdir=out --seeds=0-3 --network=https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-100-shot-obama.pkl
```

## Other Usages

To train on larger datasets (e.g., CIFAR and FFHQ), please follow the guidelines in the [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) README to prepare the datasets.

## Disclaimers

This PyTorch codebase will not fully reproduce our paper's results, as it uses a different set of hyperparameters and a different evaluation protocal. Please refer to our [TensorFlow repo](https://github.com/mit-han-lab/data-efficient-gans/tree/master/DiffAugment-stylegan2) to fully reproduce the paper's results.
