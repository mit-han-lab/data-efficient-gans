# DiffAugment for BigGAN (ImageNet)

This repo is implemented upon the [compare_gan repo](https://github.com/google/compare_gan). You will need:

- TensorFlow 1.15.
- A single GPU (for generation and evaluation).
- TPU v2/v3 Pod with 128 cores (for training).
- Manual download of the ImageNet dataset (for evaluation and training). Please follow the instructions [here](https://www.tensorflow.org/datasets/catalog/imagenet2012).
- Run `pip install -e .` to install other requirements.

## Pre-Trained Models

The following script is an example for generation using the pre-trained models:

```bash
CUDA_VISIBLE_DEVICES=0 python generate.py --tfhub_url=WHICH_MODEL
```

The generation results will be placed in the `samples` folder. Run the following command to evaluate a pre-trained model (for IS and FID):

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --tfhub_url=WHICH_MODEL
```

The evaluation typically takes around half an hour. `WHICH_MODEL` specifies the name of a pre-trained model in the following list (which will be automatically downloaded):

| Model name                                           | Dataset           | IS     | FID    |
| ---------------------------------------------------- | ----------------- | --------- | --------- |
| `mit-han-lab:biggan-imagenet128`                     | `imagenet_128`             | 9.06      | 9.59      |
| `mit-han-lab:DiffAugment-biggan-imagenet128`         | `imagenet_128`             | **9.16**  | **8.70**  |
| `mit-han-lab:biggan-imagenet128-0.5`                 | `imagenet_128` (50% data)  | 8.41      | 21.58     |
| `mit-han-lab:DiffAugment-biggan-imagenet128-0.5`     | `imagenet_128` (50% data)  | **8.65**  | **14.04** |
| `mit-han-lab:biggan-imagenet128-0.25`                 | `imagenet_128` (25% data)  | 7.62      | 39.78     |
| `mit-han-lab:DiffAugment-biggan-imagenet128-0.25`     | `imagenet_128` (25% data)  | **8.09**  | **22.40** |

The evaluation results of the pre-trained models should be close to these numbers. Specify `--num_eval_averaging_runs` to compute means and standard deviations over multiple evaluation runs. A standard deviation of less than 1% relatively is expected.

## Training

The training configs are provided in the `DiffAugment_configs` folder, which have the same naming format as the pre-trained models listed above. For example, the following command will run *BigGAN + DiffAugment* on ImageNet 128x128 with 25% training data:

```bash
TPU_NAME=YOUR_TPU_NAME python compare_gan/main.py --model_dir=MODEL_DIR --gin_config=DiffAugment_configs/DiffAugment-biggan-imagenet128-0.25.gin
```

See also the [compare_gan](https://github.com/google/compare_gan) README for more usages.
