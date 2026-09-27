# RTRM-IR Dataset1 V77 Reproduction

## Repository layout

```text
RTRM-IR/
|-- data/                   dataset loaders
|-- models/                 RTRM-IR model implementation
|-- options/                training and test options
|-- scripts/                inference and analysis scripts
|-- util/                   project utilities
|-- IQA/                    project PSNR/SSIM helpers
|-- metrics/                requested five-metric implementation
|-- checkpoints/            Dataset1 V77 best_val weights
|-- train.py                training entry point
|-- train_from_bundle.py    V77 configuration launcher
|-- evaluate_with_user_iqa.py
|-- run_train_dataset1_v77.sh
`-- run_test_dataset1_v77.sh
```

Generated files are written to `outputs/` or `reproduced_training/`; both are
ignored by Git. A local `datasets1/` directory is also ignored and will never
be uploaded.

## Environment

Create the recorded Python 3.7 environment:

```bash
conda env create -f environment-py37.yml
conda activate py3.7
python -m pip install -r requirements-py37.txt
```

The verified environment uses Python 3.7.1, PyTorch 1.13.1, and CUDA 11.7.

## Dataset

Download `datasets1 (ODinMJ-part)` from Baidu Netdisk:

- Download link: [https://pan.baidu.com/s/1l0jV0KIbeoMeq7xLhy4FNQ?pwd=2evh](https://pan.baidu.com/s/1l0jV0KIbeoMeq7xLhy4FNQ?pwd=2evh)
- Extraction code: `2evh`

After downloading, extract the archive and place or rename the dataset folder
as `datasets1/` in the repository root. The dataset is distributed separately
and is not tracked by Git.

Place the dataset locally at `datasets1/`:

```text
datasets1/
|-- trainA/
|-- trainB/
|-- trainGT/
|-- testA/
|-- testB/
`-- testGT/
```

The test route requires `testA`, `testB`, and `testGT`. A different location
can be supplied with `DATASET_ROOT`.

## Reproduce the test

From the repository root:

```bash
conda activate py3.7
./run_test_dataset1_v77.sh
```

Example with explicit paths:

```bash
DATASET_ROOT=/path/to/datasets1 GPU_ID=0 ./run_test_dataset1_v77.sh
```

