# DiffAugment for StyleGAN2 (PyTorch)

This repo is implemented upon [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) with minimal modifications to train and load DiffAugment-stylegan2 models in PyTorch. Please check the [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) README for the dependencies and the other usages of this codebase.

## Requirements

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory. We have done all testing and development using NVIDIA DGX-1 with 8 Tesla V100 GPUs.
* 64-bit Python 3.7 and PyTorch 1.7.1. See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.
* CUDA toolkit 11.0 or later.  Use at least version 11.1 if running on RTX 3090.  (Why is a separate CUDA toolkit installation required?  See comments in [#2](https://github.com/NVlabs/stylegan2-ada-pytorch/issues/2#issuecomment-779457121).)
* Python libraries: `pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3`.  We use the Anaconda3 2020.11 distribution which installs most of these by default.
* Docker users: use the [provided Dockerfile](./Dockerfile) to build an image with the required library dependencies.

The code relies heavily on custom PyTorch extensions that are compiled on the fly using NVCC. On Windows, the compilation requires Microsoft Visual Studio. We recommend installing [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/) and adding it into `PATH` using `"C:\Program Files (x86)\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvars64.bat"`.

## Low-Shot Generation

The following command is an example of training StyleGAN2 with the default *Color + Translation + Cutout* DiffAugment on 100-shot Obama with 1 GPU. See [here](https://data-efficient-gans.mit.edu/datasets/) for a list of our provided low-shot datasets. You may also prepare your own dataset and specify the path to your image folder.
```bash
python train.py --outdir=training-runs --data=https://data-efficient-gans.mit.edu/datasets/100-shot-obama.zip --gpus=1
```

## Pre-Trained Models

The following commands are an example of generating images with our pre-trained 100-shot Obama model. See [here](https://data-efficient-gans.mit.edu/models/) for a list of our provided pre-trained models. The code will automatically convert a TensorFlow StyleGAN2 model to the compatible PyTorch version; you may also use `legacy.py` to do this manually.
```bash
python generate.py --outdir=out --seeds=1-16 --network=https://data-efficient-gans.mit.edu/models/DiffAugment-stylegan2-100-shot-obama.pkl

python generate_gif.py --output=obama.gif --seed=0 --num-rows=1 --num-cols=8 --network=https://data-efficient-gans.mit.edu/models/DiffAugment-stylegan2-100-shot-obama.pkl
```

<img src="../imgs/obama.gif" width="1000px"/>

## Other Usages

To train on larger datasets (e.g., CIFAR and FFHQ), please follow the guidelines in the [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) README to prepare the datasets.

## Disclaimer

This PyTorch codebase will not fully reproduce our paper's results, as it uses a different set of hyperparameters and a different evaluation protocal. Please refer to our [TensorFlow repo](https://github.com/mit-han-lab/data-efficient-gans/tree/master/DiffAugment-stylegan2) to fully reproduce the paper's results.
