# DiffAugment for StyleGAN2

This repo is implemented upon and has the same dependencies as the official [StyleGAN2 repo](https://github.com/NVlabs/stylegan2). Specifically,

- TensorFlow 1.14 or 1.15 with GPU support.
- `tensorflow-datasets` version <= 2.1.0 should be installed to run on CIFAR, e.g., `pip install tensorflow-datasets==2.1.0`.
- We recommend using 4 or 8 GPUs with at least 12 GB of DRAM for training.
- If you are facing problems with `nvcc` (when building custom ops of StyleGAN2), this can be circumvented by specifying `--impl=ref` in training at the cost of a slightly longer training time.

## Generating an Interpolation Video

<img src="../imgs/interp.gif"/>

Run the following command to generate an interpolation video:

```bash
python generate_gif.py --resume=WHICH_MODEL --output=OUTPUT_FILENAME
```

`WHICH_MODEL` specifies the path of a checkpoint or any pre-trained models in the tables below, e.g., `mit-han-lab:DiffAugment-stylegan2-100-shot-obama.pkl`.

## CIFAR-10 and CIFAR-100

To run the CIFAR experiments with 100% data:

```bash
python run_cifar.py --dataset=WHICH_DATASET --num-gpus=NUM_GPUS --DiffAugment=color,cutout
```

`WHICH_DATASET` specifies either `cifar10` or `cifar100` (defaults to `cifar10`). `NUM_GPUS` specifies the number of GPUs to use; we recommend using 4 or 8 GPUs to replicate our results. The training typically takes around 2 days. Set `--DiffAugment=""` to run the baseline model.

To run the CIFAR experiments with partial data:

```bash
python run_cifar.py --dataset=WHICH_DATASET --num-samples=NUM_SAMPLES --num-gpus=NUM_GPUS --DiffAugment=color,translation,cutout
```

`WHICH_DATASET` specifies either `cifar10` or `cifar100` (defaults to `cifar10`). `NUM_SAMPLES` specifies the number of training samples to use, e.g., `5000` for 10% data or `10000` for 20% data. `NUM_GPUS` specifies the number of GPUs to use; we recommend using 4 or 8 GPUs to replicate our results. Set `--DiffAugment=""` to run the baseline model.

### Pre-Trained Models and Evaluation

To evaluate a model on CIFAR-10 or CIFAR-100, run the following command:

```bash
python run_cifar.py --dataset=WHICH_DATASET --resume=WHICH_MODEL --eval
```

Here, `WHICH_DATASET` specifies either `cifar10` or `cifar100` (defaults to `cifar10`); `WHICH_MODEL` specifies the path of a checkpoint, or a pre-trained model in the following list, which will be automatically downloaded:

| Model name | Dataset | is10k | fid10k |
| --- | --- | --- | --- |
| `mit-han-lab:stylegan2-cifar10.pkl` | `cifar10` | 9.18 | 11.07 |
| `mit-han-lab:DiffAugment-stylegan2-cifar10.pkl` | `cifar10` | **9.40** | **9.89** |
| `mit-han-lab:stylegan2-cifar10-0.2.pkl` | `cifar10` (20% data) | 8.28 | 23.08 |
| `mit-han-lab:DiffAugment-stylegan2-cifar10-0.2.pkl` | `cifar10` (20% data) | **9.21** | **12.15** |
| `mit-han-lab:stylegan2-cifar10-0.1.pkl` | `cifar10` (10% data) | 7.33 | 36.02 |
| `mit-han-lab:DiffAugment-stylegan2-cifar10-0.1.pkl` | `cifar10` (10% data) | **8.84** | **14.50** |
| `mit-han-lab:stylegan2-cifar100.pkl` | `cifar100` | 9.51 | 16.54 |
| `mit-han-lab:DiffAugment-stylegan2-cifar100.pkl` | `cifar100` | **10.04** | **15.22** |
| `mit-han-lab:stylegan2-cifar100-0.2.pkl` | `cifar100` (20% data) | 7.86 | 32.30 |
| `mit-han-lab:DiffAugment-stylegan2-cifar100-0.2.pkl` | `cifar100` (20% data) | **9.82** | **16.65** |
| `mit-han-lab:stylegan2-cifar100-0.1.pkl` | `cifar100` (10% data) | 7.01 | 45.87 |
| `mit-han-lab:DiffAugment-stylegan2-cifar100-0.1.pkl` | `cifar100` (10% data) | **9.06** | **20.75** |

The evaluation results of the pre-trained models should be close to these numbers. Specify `--num-repeats=REPEATS` to compute means and standard deviations over multiple evaluation runs. A standard deviation of less than 1% relatively is expected.

## FFHQ and LSUN

The NVIDIA's FFHQ dataset can be downloaded [here](https://drive.google.com/drive/folders/1M24jfI-Ylb-k2EGhELSnxssWi9wGUokg). If you want to run at 256x256 resolution for example, only `ffhq-r08.tfrecords` needs to be downloaded. The LSUN datasets (in LMDB format) can be downloaded [here](https://www.yf.io/p/lsun). Pass the folder containing the `.tfrecords` or `.mdb` file to `PATH_TO_THE_TFRECORDS_OR_LMDB_FOLDER` below:

```bash
python run_ffhq.py --dataset=PATH_TO_THE_TFRECORDS_OR_LMDB_FOLDER --num-samples=NUM_SAMPLES --num-gpus=NUM_GPUS --resolution=256 --DiffAugment=color,translation,cutout
```

If there are multiple `.tfrecords` files in the folder, the one with the highest resolution will be used.

### Pre-Trained Models and Evaluation

Run the following command to evaluate a model on the FFHQ/LSUN dataset:

```bash
python run_ffhq.py --dataset=PATH_TO_THE_TFRECORDS_OR_LMDB_FOLDER --resume=WHICH_MODEL --num-gpus=NUM_GPUS --eval
```

Here, `PATH_TO_THE_TFRECORDS_OR_LMDB_FOLDER` specifies the folder containing the `.tfrecords` or `.mdb` file. `WHICH_MODEL` specifies the path of a checkpoint, or a pre-trained model in the list below, which will be automatically downloaded. The pre-trained models are run at 256x256 resolution using 8 GPUs. We apply the strongest *Color + Translation + Cutout* DiffAugment to all the baselines, which significantly gains the performance when training with partial data:

| Model name                                                  | Dataset               | fid50k-train |
| ----------------------------------------------------------- | --------------------- | ----------- |
| `mit-han-lab:stylegan2-ffhq.pkl`                | FFHQ (full, 70k samples) | **3.81** |
| `mit-han-lab:DiffAugment-stylegan2-ffhq.pkl`    | FFHQ (full, 70k samples) | 4.24 |
| `mit-han-lab:stylegan2-ffhq-30k.pkl`            | FFHQ (30k samples)  | 6.16 |
| `mit-han-lab:DiffAugment-stylegan2-ffhq-30k.pkl` | FFHQ (30k samples)  | **5.05** |
| `mit-han-lab:stylegan2-ffhq-10k.pkl`            | FFHQ (10k samples)  | 14.75 |
| `mit-han-lab:DiffAugment-stylegan2-ffhq-10k.pkl` | FFHQ (10k samples)  | **7.86** |
| `mit-han-lab:stylegan2-ffhq-5k.pkl`             | FFHQ (5k samples)   | 26.60 |
| `mit-han-lab:DiffAugment-stylegan2-ffhq-5k.pkl` | FFHQ (5k samples)   | **10.45** |
| `mit-han-lab:stylegan2-ffhq-1k.pkl`             | FFHQ (1k samples)   | 62.16 |
| `mit-han-lab:DiffAugment-stylegan2-ffhq-1k.pkl` | FFHQ (1k samples)   | **25.66** |
| `mit-han-lab:stylegan2-lsun-cat-30k.pkl`            | LSUN-Cat (30k samples)  | 10.12 |
| `mit-han-lab:DiffAugment-stylegan2-lsun-cat-30k.pkl` | LSUN-Cat (30k samples)  | **9.68** |
| `mit-han-lab:stylegan2-lsun-cat-10k.pkl`            | LSUN-Cat (10k samples)  | 17.93 |
| `mit-han-lab:DiffAugment-stylegan2-lsun-cat-10k.pkl` | LSUN-Cat (10k samples)  | **12.07** |
| `mit-han-lab:stylegan2-lsun-cat-5k.pkl`             | LSUN-Cat (5k samples)   | 34.69 |
| `mit-han-lab:DiffAugment-stylegan2-lsun-cat-5k.pkl` | LSUN-Cat (5k samples)   | **16.11** |
| `mit-han-lab:stylegan2-lsun-cat-1k.pkl`             | LSUN-Cat (1k samples)   | 182.85 |
| `mit-han-lab:DiffAugment-stylegan2-lsun-cat-1k.pkl` | LSUN-Cat (1k samples)   | **42.26** |

## Low-Shot Generation

<img src="../imgs/low-shot-interp.jpg" width="1000px"/>

To run the low-shot generation experiments on the 100-shot datasets:

```bash
python run_low_shot.py --dataset=WHICH_DATASET --num-gpus=NUM_GPUS --DiffAugment=color,translation,cutout
```

or the following command to run on the AnimalFace datasets (with a longer training length):

```bash
python run_low_shot.py --dataset=WHICH_DATASET --num-gpus=NUM_GPUS --DiffAugment=color,translation,cutout --total-kimg=500
```

`WHICH_DATASET` specifies `100-shot-obama`, `100-shot-grumpy_cat`, `100-shot-panda`, `100-shot-bridge_of_sighs`, `100-shot-medici_fountain`, `100-shot-temple_of_heaven`, `100-shot-wuzhen`, `AnimalFace-cat`, or `AnimalFace-dog`, which will be automatically downloaded, or the path of a folder containing your own training images. `NUM_GPUS` specifies the number of GPUs to use; we recommend using 4 or 8 GPUs to replicate our results. The training typically takes several hours. Set `--DiffAugment=""` to run the baseline model. Specify `--resolution=RESOLUTION` to run at a different resolution from the default `256`. You may also fine-tune from an FFHQ pre-trained model listed above, e.g., by specifying `--resume=mit-han-lab:DiffAugment-stylegan2-ffhq.pkl --fmap-base=8192`.

### Preparing Your Own Datasets

Our method can generate good results using a small number of samples, e.g., 100 images. You may create a new dataset at such scale easily, but note that the generated results may be sensitive to the quality of the training samples. You may wish to crop the raw images and discard some bad training samples. After putting all images into a single folder, pass it to `WHICH_DATASET` in `run_low_shot.py`, the images will be resized to the specified resolution if necessary, and then enjoy the outputs! Note that,

- The training length (defaults to 300k images) may be increased for larger datasets, but there may be overfitting issues if the training is too long.
- The cached files will be stored in the same folder with the training images. If the training images in your folder is *changed* after some run, please manually clean the cached files, `*.tfrecords` and `*.pkl`, from your image folder before rerun.

### Pre-Trained Models and Evaluation

To evaluate a model on a low-shot generation dataset, run the following command:

```bash
python run_low_shot.py --dataset=WHICH_DATASET --resume=WHICH_MODEL --eval
```

Here, `WHICH_DATASET` specifies the folder containing the training images, or one of our pre-defined datasets, including `100-shot-obama`, `100-shot-grumpy_cat`, `100-shot-panda`, `100-shot-bridge_of_sighs`, `100-shot-medici_fountain`, `100-shot-temple_of_heaven`, `100-shot-wuzhen`, `AnimalFace-cat`, and `AnimalFace-dog`, which will be automatically downloaded. `WHICH_MODEL` specifies the path of a checkpoint, or a pre-trained model in the following list, which will be automatically downloaded:
| Model name | Dataset | fid5k-train |
| --- | --- | --- |
| `mit-han-lab:stylegan2-100-shot-obama.pkl` | `100-shot-obama` | 80.20 |
| `mit-han-lab:DiffAugment-stylegan2-100-shot-obama.pkl` | `100-shot-obama` | **46.87** |
| `mit-han-lab:stylegan2-100-shot-grumpy_cat.pkl` | `100-shot-grumpy_cat` | 48.90 |
| `mit-han-lab:DiffAugment-stylegan2-100-shot-grumpy_cat.pkl` | `100-shot-grumpy_cat` | **27.08** |
| `mit-han-lab:stylegan2-100-shot-panda.pkl` | `100-shot-panda` | 34.27 |
| `mit-han-lab:DiffAugment-stylegan2-100-shot-panda.pkl` | `100-shot-panda` | **12.06** |
| `mit-han-lab:stylegan2-AnimalFace-cat.pkl` | `AnimalFace-cat` | 71.71 |
| `mit-han-lab:DiffAugment-stylegan2-AnimalFace-cat.pkl` | `AnimalFace-cat` | **42.44** |
| `mit-han-lab:stylegan2-AnimalFace-dog.pkl` | `AnimalFace-dog` | 130.19 |
| `mit-han-lab:DiffAugment-stylegan2-AnimalFace-dog.pkl` | `AnimalFace-dog` | **58.85** |
