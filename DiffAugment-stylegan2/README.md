# StyleGAN2 + DiffAugment

This repo is implemented upon and has the same dependencies as the official [StyleGAN2 repo](https://github.com/NVlabs/stylegan2).

## Pre-trained Models and Evaluation

To evaluate a model on CIFAR-10 or CIFAR-100, run the following command:

```bash
python run_cifar.py --dataset=WHICH_DATASET --resume=WHICH_MODEL --eval
```

Here, `WHICH_DATASET` specifies either `cifar10` or `cifar100` (default to `cifar10`); `WHICH_MODEL` specifies the path of a checkpoint, or a pre-trained model in the following list, which will be automatically downloaded:

`mit-han-lab:stylegan2-cifar10.pkl`<br>
`mit-han-lab:stylegan2-cifar10-0.1.pkl`<br>
`mit-han-lab:stylegan2-cifar10-0.2.pkl`<br>
`mit-han-lab:stylegan2-cifar100.pkl`<br>
`mit-han-lab:stylegan2-cifar100-0.1.pkl`<br>
`mit-han-lab:stylegan2-cifar100-0.2.pkl`<br>
`mit-han-lab:DiffAugment-stylegan2-cifar10.pkl`<br>
`mit-han-lab:DiffAugment-stylegan2-cifar10-0.1.pkl`<br>
`mit-han-lab:DiffAugment-stylegan2-cifar10-0.2.pkl`<br>
`mit-han-lab:DiffAugment-stylegan2-cifar100.pkl`<br>
`mit-han-lab:DiffAugment-stylegan2-cifar100-0.1.pkl`<br>
`mit-han-lab:DiffAugment-stylegan2-cifar100-0.2.pkl`

To evaluate a model on a few-shot dataset, run the following command:

```bash
python run_few_shot.py --dataset=WHICH_DATASET --resume=WHICH_MODEL --eval
```

Here, `WHICH_DATASET` specifies the folder containing the training images, or one of our 100-shot datasets, `100-shot-obama`, `100-shot-grumpy_cat`, or `100-shot-panda`, which will be automatically downloaded. `WHICH_MODEL` specifies the path of a checkpoint. Pre-trained models for few-shot generation is coming!

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

To run the few-shot generation experiments at 256×256 resolution:

```bash
python run_few_shot.py --dataset=WHICH_DATASET --num-gpus=NUM_GPUS --DiffAugment=color,translation,cutout
```

`WHICH_DATASET` specifies `100-shot-obama`, `100-shot-grumpy_cat`, or `100-shot-panda` which will be automatically downloaded, or the path of a folder containing your own training images. `NUM_GPUS` specifies the number of GPUs to use; we recommend using 4 or 8 GPUs to replicate our results.
