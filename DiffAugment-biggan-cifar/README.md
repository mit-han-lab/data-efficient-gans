# DiffAugment for BigGAN (CIFAR)

This repo is implemented upon the [BigGAN-PyTorch repo](https://github.com/ajbrock/BigGAN-PyTorch). The main dependencies are:

- PyTorch version >= 1.0.1. Code has been tested with PyTorch 1.4.0.

- TensorFlow 1.14 or 1.15 with GPU support (for IS and FID calculation).

- We recommend using 2 GPUs with at least 12 GB of DRAM for training and evaluation.

## Pre-Trained Models and Evaluation

To evaluate a model on CIFAR-10 or CIFAR-100, run the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python eval.py --dataset=WHICH_DATASET --network=WHICH_MODEL
```

Here, `WHICH_DATASET` specifies either `C10` (CIFAR-10) or `C100` (CIFAR-100), default to `C10`; `WHICH_MODEL` specifies the path of a checkpoint containing the generator's weights (typically this should be the path of a file named `G_ema_best.pth` in the `weights` folder), or a pre-trained model in the following list, which will be automatically downloaded:

| Model name                                           | Dataset               | is10k     | fid10k    |
| ---------------------------------------------------- | --------------------- | --------- | --------- |
| `mit-han-lab:biggan-cifar10.pth`                     | `C10`             | 9.06      | 9.59      |
| `mit-han-lab:DiffAugment-biggan-cifar10.pth`         | `C10`             | **9.16**  | **8.70**  |
| `mit-han-lab:cr-biggan-cifar10.pth`                  | `C10`             | **9.20**  | 9.06      |
| `mit-han-lab:DiffAugment-cr-biggan-cifar10.pth`      | `C10`             | 9.17      | **8.49**  |
| `mit-han-lab:biggan-cifar10-0.2.pth`                 | `C10` (20% data)  | 8.41      | 21.58     |
| `mit-han-lab:DiffAugment-biggan-cifar10-0.2.pth`     | `C10` (20% data)  | **8.65**  | **14.04** |
| `mit-han-lab:cr-biggan-cifar10-0.2.pth`              | `C10` (20% data)  | 8.43      | 20.62     |
| `mit-han-lab:DiffAugment-cr-biggan-cifar10-0.2.pth`  | `C10` (20% data)  | **8.61**  | **12.84** |
| `mit-han-lab:biggan-cifar10-0.1.pth`                 | `C10` (10% data)  | 7.62      | 39.78     |
| `mit-han-lab:DiffAugment-biggan-cifar10-0.1.pth`     | `C10` (10% data)  | **8.09**  | **22.40** |
| `mit-han-lab:cr-biggan-cifar10-0.1.pth`              | `C10` (10% data)  | 7.66      | 37.45     |
| `mit-han-lab:DiffAugment-cr-biggan-cifar10-0.1.pth`  | `C10` (10% data)  | **8.49**  | **18.70** |
| `mit-han-lab:biggan-cifar100.pth`                    | `C100`            | **10.92** | 12.87     |
| `mit-han-lab:DiffAugment-biggan-cifar10.pth`         | `C100`            | 10.66     | **12.00** |
| `mit-han-lab:cr-biggan-cifar100.pth`                 | `C100`            | **10.95** | 11.26     |
| `mit-han-lab:DiffAugment-cr-biggan-cifar10.pth`      | `C100`            | 10.81     | **11.25** |
| `mit-han-lab:biggan-cifar100-0.2.pth`                | `C100` (20% data) | 9.11      | 33.11     |
| `mit-han-lab:DiffAugment-biggan-cifar100-0.2.pth`    | `C100` (20% data) | **9.47**  | **22.14** |
| `mit-han-lab:cr-biggan-cifar100-0.2.pth`             | `C100` (20% data) | 8.44      | 36.91     |
| `mit-han-lab:DiffAugment-cr-biggan-cifar100-0.2.pth` | `C100` (20% data) | **9.12**  | **20.28** |
| `mit-han-lab:biggan-cifar100-0.1.pth`                | `C100` (10% data) | 5.94      | 66.71     |
| `mit-han-lab:DiffAugment-biggan-cifar100-0.1.pth`    | `C100` (10% data) | **8.38**  | **33.70** |
| `mit-han-lab:cr-biggan-cifar100-0.1.pth`             | `C100` (10% data) | 7.91      | 47.16     |
| `mit-han-lab:DiffAugment-cr-biggan-cifar100-0.1.pth` | `C100` (10% data) | **8.70**  | **26.90** |

The evaluation results of the pre-trained models should be close to these numbers. Specify `--repeat=NUM_REPEATS` to compute means and standard deviations over multiple evaluation runs. A standard deviation of less than 1% relatively is expected.

## Training

We provide a complete set of training scripts in the `scripts` folder to facilitate replicating our results. The scripts have the same naming format as the pre-trained models listed above. For example, the following command will run *BigGAN + DiffAugment* on CIFAR-10 with 10% training data:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/DiffAugment-biggan-cifar10-0.1.sh
```

The training typically requires around 1 day on 2 GPUs.

## Acknowledgements

The official TensorFlow implementation of the Inception v3 model for IS and FID calculation is borrowed from the [StyleGAN2 repo](https://github.com/NVlabs/stylegan2).