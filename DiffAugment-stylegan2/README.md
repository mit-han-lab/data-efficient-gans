# DiffAugment for StyleGAN2

This repo is implemented upon and has the same dependencies as the official [StyleGAN2 repo](https://github.com/NVlabs/stylegan2). Specifically,

- TensorFlow 1.14 or 1.15 with GPU support.
- `tensorflow-datasets 2.1.0` should be installed to run on CIFAR, e.g., `pip install tensorflow-datasets==2.1.0`.
- We recommend using 4 or 8 GPUs with at least 12 GB of DRAM for training.
- If you are facing problems with `nvcc` (when building custom ops of StyleGAN2), this can be circumvented by specifying `--impl=ref` in training at the cost of a slightly longer training time.

## Pre-Trained Models and Evaluation

To evaluate a model on CIFAR-10 or CIFAR-100, run the following command:

```bash
python run_cifar.py --dataset=WHICH_DATASET --resume=WHICH_MODEL --eval
```

Here, `WHICH_DATASET` specifies either `cifar10` or `cifar100` (default to `cifar10`); `WHICH_MODEL` specifies the path of a checkpoint, or a pre-trained model in the following list, which will be automatically downloaded:

| Model name | Dataset | IS | FID |
| --- | --- | --- | --- |
| `mit-han-lab:stylegan2-cifar10.pkl` | `cifar10` | 9.18 | 11.07 |
| `mit-han-lab:DiffAugment-stylegan2-cifar10.pkl` | `cifar10` | **9.40** | **9.89** |
| `mit-han-lab:stylegan2-cifar10-0.2.pkl` | `cifar10` (20% data) | 8.28 | 23.08 |
| `mit-han-lab:DiffAugment-stylegan2-cifar10-0.2.pkl` | `cifar10` (20% data) | **9.21** | **12.15** |
| `mit-han-lab:stylegan2-cifar10-0.1.pkl` | `cifar10` (10% data) | 7.33 | 36.02 |
| `mit-han-lab:DiffAugment-stylegan2-cifar10-0.1.pkl` | `cifar10` (10% data) | **8.84** | **14.50** |
| `mit-han-lab:stylegan2-cifar100.pkl` | `cifar100` | 9.51 | 16.54 |
| `mit-han-lab:DiffAugment-stylegan2-cifar10.pkl` | `cifar100` | **10.04** | **15.22** |
| `mit-han-lab:stylegan2-cifar100-0.2.pkl` | `cifar100` (20% data) | 7.86 | 32.30 |
| `mit-han-lab:DiffAugment-stylegan2-cifar100-0.2.pkl` | `cifar100` (20% data) | **9.82** | **16.65** |
| `mit-han-lab:stylegan2-cifar100-0.1.pkl` | `cifar100` (10% data) | 7.01 | 45.87 |
| `mit-han-lab:DiffAugment-stylegan2-cifar100-0.1.pkl` | `cifar100` (10% data) | **9.06** | **20.75** |

The evaluation results of the pre-trained models should be close to these numbers.

To evaluate a model on a few-shot dataset, run the following command:

```bash
python run_few_shot.py --dataset=WHICH_DATASET --resume=WHICH_MODEL --eval
```

Here, `WHICH_DATASET` specifies the folder containing the training images, or one of pre-defined datasets, including `100-shot-obama`, `100-shot-grumpy_cat`, `100-shot-panda`, `AnimalFace-cat`, and `AnimalFace-dog`, which will be automatically downloaded. `WHICH_MODEL` specifies the path of a checkpoint, or a pre-trained model in the following list, which will be automatically downloaded:
| Model name | Dataset | FID |
| --- | --- | --- |
| `mit-han-lab:stylegan2-100-shot-obama.pkl` | `100-shot-obama` | 89.18 |
| `mit-han-lab:DiffAugment-stylegan2-100-shot-obama.pkl` | `100-shot-obama` | **54.39** |
| `mit-han-lab:stylegan2-100-shot-grumpy_cat.pkl` | `100-shot-grumpy_cat` | 61.97 |
| `mit-han-lab:DiffAugment-stylegan2-100-shot-grumpy_cat.pkl` | `100-shot-grumpy_cat` | **29.90** |
| `mit-han-lab:stylegan2-100-shot-panda.pkl` | `100-shot-panda` | 90.96 |
| `mit-han-lab:DiffAugment-stylegan2-100-shot-panda.pkl` | `100-shot-panda` | **13.21** |
| `mit-han-lab:stylegan2-AnimalFace-cat.pkl` | `AnimalFace-cat` | 95.75 |
| `mit-han-lab:DiffAugment-stylegan2-AnimalFace-cat.pkl` | `AnimalFace-cat` | **46.51** |
| `mit-han-lab:stylegan2-AnimalFace-dog.pkl` | `AnimalFace-dog` | 164.54 |
| `mit-han-lab:DiffAugment-stylegan2-AnimalFace-dog.pkl` | `AnimalFace-dog` | **62.78** |

## Training

### CIFAR-10 and CIFAR-100

To run the CIFAR experiments with 100% data:

```bash
python run_cifar.py --dataset=WHICH_DATASET --num-gpus=NUM_GPUS --DiffAugment=color,cutout
```

`WHICH_DATASET` specifies either `cifar10` or `cifar100` (default to `cifar10`). `NUM_GPUS` specifies the number of GPUs to use; we recommend using 4 or 8 GPUs to replicate our results. Set `--DiffAugment=""` to run the baseline model.

To run the CIFAR experiments with partial data:

```bash
python run_cifar.py --dataset=WHICH_DATASET --num-samples=NUM_SAMPLES --num-gpus=NUM_GPUS --DiffAugment=color,translation,cutout
```

`WHICH_DATASET` specifies either `cifar10` or `cifar100` (default to `cifar10`). `NUM_SAMPLES` specifies the number of training samples to use, e.g., `5000` for 10% data or `10000` for 20% data. `NUM_GPUS` specifies the number of GPUs to use; we recommend using 4 or 8 GPUs to replicate our results. Set `--DiffAugment=""` to run the baseline model.

### Few-Shot Generation

To run the few-shot generation experiments:

```bash
python run_few_shot.py --dataset=WHICH_DATASET --num-gpus=NUM_GPUS --DiffAugment=color,translation,cutout
```

`WHICH_DATASET` specifies `100-shot-obama`, `100-shot-grumpy_cat`, `100-shot-panda`, `AnimalFace-cat`, or `AnimalFace-dog` which will be automatically downloaded, or the path of a folder containing your own training images. `NUM_GPUS` specifies the number of GPUs to use; we recommend using 4 or 8 GPUs to replicate our results. Specify `--resolution=RESOLUTION` to run at a different resolution from the default `256`.

### Preparing Your Own Datasets

Our method can generate good results using a small number of samples, e.g., 100 images. You may create a new dataset at such scale easily, but note that the generated results may be sensitive to the quality of the training samples. You may wish to crop the raw images and discard some bad training samples. After putting all images into a single folder, pass it to `WHICH_DATASET`, the images will be resized to the specified resolution if necessary, and then enjoy the outputs!