# Dataset1 V77 Test Reproduction

This bundle is for the **dataset1 V77 checkpoint on the dataset1 test split**.
It does not contain the dataset.

## Fixed experiment

- Checkpoint: `best_val`
- Experiment:
  `datasets1_V77_guidedK19_TirBlur31_cosPrompt_fusionRes1_scratch_bs4_ep520_gpu6_20260809_054006`
- Best validation epoch: `454`
- Test split: `dataset1/testA`, paired with `testB` and `testGT` when computing metrics
- Number of test images: `160`
- Original image size is preserved by the no-reference inference route.

The bundled checkpoint files are the `best_val_net_*.pth` files. The dataset is
intentionally not included. When the script is run from this bundle, the
default dataset path is the relative path:

```text
../datasets1
```

Override the location without changing the bundle:

```bash
DATASET_ROOT=../datasets1 ./run_test_dataset1_v77.sh
```

The runner uses `python` from the active environment by default:

```text
python
```

Override it with `PYTHON_BIN=/path/to/python` when the environment is not active.

## Python environment

Create the bundled Python 3.7 environment from the repository root:

```bash
conda env create -f environment-py37.yml
conda activate py3.7
```

For an existing Python 3.7 environment, install the pinned packages with:

```bash
python -m pip install -r requirements-py37.txt
```

The recorded environment uses Python 3.7.1, PyTorch 1.13.1 with CUDA 11.7,
and the dependency versions listed in `requirements-py37.txt`.

## Historical V77 source

The executable source tree for this reproduction is `historical_v77_source/`.
The runner sets `PYTHONPATH` to this directory. The preserved historical output
files and metric record are the target reference:

```text
PSNR: 32.485579151761236
SSIM: 0.83698733726667
AG: 3.2881651675561683
EME: 2.458763262407466
NIQE: 18.918707629932836
```

The source recovery and code checks are recorded in
`test_metadata/HISTORICAL_V77_SOURCE_RECOVERY.md`. The later working-tree
snapshot is retained under
`test_metadata/current_source_audit/source_snapshot/` for audit only and is
not used by the runner.

## Test and evaluation

The historical V77 test record used the paired full-size route represented by
`historical_v77_source/scripts/save_paired_test_outputs.py`. It keeps the training
forward state and preserves the original image size. The preserved historical
outputs are in:

```text
results/historical_dataset1_testA/
```

The fresh paired reproduction output is written to
`results/historical_source_paired_dataset1_testA/`.

The final PSNR/SSIM evaluation uses the requested implementation:

```text
metric_reference/IQA我自己的/IQA/psnr_ssim.py
```

The wrapper `evaluate_with_user_iqa.py` follows
`IQA我自己的/IQA/test.py`: `cv2.imread`, sorted image pairing, and
`crop_border=4`. It reports all requested metrics:

```text
PSNR ↑, SSIM ↑, AG ↑, EME ↑, NIQE ↓
```

AG, EME, and NIQE are computed from the result image converted to grayscale,
matching `IQA我自己的/IQA/test.py`. The result is saved under
`results/metrics/` as JSON, CSV, and a log file. PSNR/SSIM require `testGT`;
AG/EME/NIQE do not require a reference image.

## Reproduction

From this directory:

```bash
./run_test_dataset1_v77.sh
```

The script runs the paired full-size route from `historical_v77_source` on one visible GPU. It maps
the selected physical GPU to logical device `0` using `CUDA_VISIBLE_DEVICES`,
so a command such as `GPU_ID=0` means physical GPU 0. The historical output
images and historical metric record are preserved separately and are not
overwritten by the reproduction run.

The historical record remains documented in
`test_metadata/HISTORICAL_DATASET1_V77_TEST.md`. New runs use the requested
IQA implementation and must be read from `results/metrics/user_iqa_*.json`.

For the preserved historical output PNGs, the requested IQA implementation
currently reports:

```text
PSNR: 32.485579151761236
SSIM: 0.83698733726667
AG: 3.2881651675561683
EME: 2.458763262407466
NIQE: 18.918707629932836
```

The complete table is in `results/metrics/metrics_summary.md`. Lower NIQE is
better.

## Training source

The corresponding historical training entry point is
`historical_v77_source/train.py`. The launcher
`run_train_dataset1_v77.sh` reads the bundled checkpoint
`train_opt.json`, applies only runtime dataset, output, and GPU paths, and
launches that training entry point. It defaults to 520 epochs and writes new
training output under `reproduced_training/`, leaving the historical weights
unchanged.

Validate the full V77 option set without starting training:

```bash
PYTHON_BIN=python ./train_from_bundle.py --dry-run
```

Start a scratch reproduction:

```bash
DATASET_ROOT=../datasets1 GPU_ID=0 ./run_train_dataset1_v77.sh
```

The historical training configuration has `deterministic=0`, so different GPU
kernel choices can cause small numerical differences. The exact historical
output PNGs, checkpoint, options, command, metric record, and executable source
tree are retained in this bundle.

## Bundle layout

```text
checkpoint/                 best_val weights and training options
results/historical_source_paired_dataset1_testA/
results/metrics/
historical_v77_source/      source files used by the test routes
train_from_bundle.py        V77 option loader and training launcher
run_train_dataset1_v77.sh   relative-path training entry point
test_metadata/current_source_audit/  later snapshot, audit only
test_metadata/              hashes, historical record, and environment notes
run_test_dataset1_v77.sh
```
