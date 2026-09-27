# Historical V77 Source Recovery

The recovered executable V77 source is now the repository's root source tree:
`data/`, `models/`, `options/`, `scripts/`, `util/`, `IQA/`, and `train.py`.
The checkpoint is stored under `checkpoints/`, and the requested metric
implementation is stored under `metrics/`.

The dataset remains outside Git and is supplied at runtime. The verified
reference result is:

```text
PSNR  32.485579151761236
SSIM   0.836987337266670
AG     3.2881651675561683
EME    2.458763262407466
NIQE  18.918707629932836
```

`source.sha256`, `metrics.sha256`, `checkpoints.sha256`, and
`reference_outputs.sha256` record the files used by this repository layout.
The historical training command and original training log are retained in
this directory.

Run the verified route from the repository root:

```bash
DATASET_ROOT=/path/to/datasets1 GPU_ID=0 ./run_test_dataset1_v77.sh
```

Fresh output is written to `outputs/`; the preserved reference files under
`reference/` are not overwritten.
