# Historical V77 Source Recovery

This bundle is self-contained for the executable test path. The active source
tree is `../historical_v77_source`, the checkpoint is under `../checkpoint`,
and the metric implementation is under
`../metric_reference/IQA我自己的/IQA`. The runner receives the dataset path at
runtime; the dataset itself is intentionally outside the bundle.

The preserved target result is the existing dataset1 V77 historical output:

```text
PSNR  32.485579151761236
SSIM   0.83698733726667
AG     3.2881651675561683
EME    2.458763262407466
NIQE  18.918707629932836
```

The model source files in `historical_v77_source/models/` were checked against
the preserved historical bytecode in the audit directory. The checked files
include `Irenhance_model.py`, `HCCM.py`, `decom_net.py`, and `fusion.py`.
The source tree also includes the data loading, option, utility, and test
scripts used by the forward path, together with the historical `train.py`
entry point. `run_train_dataset1_v77.sh` loads the bundled V77 options and
starts that entry point. `historical_source.sha256` records the current hashes
of the bundled Python and shell source files.

A later working-tree snapshot is retained under
`current_source_audit/source_snapshot/` for comparison only. It is not on the
runner's `PYTHONPATH`.

Run the reproduction from the bundle directory:

```bash
cd github_dataset1_V77_test_repro
DATASET_ROOT=../datasets1 ./run_test_dataset1_v77.sh
```

The test script uses the source and IQA code from this directory. It does not
read the project root source tree. Fresh outputs go to
`results/historical_source_inference_dataset1_testA/` and
`results/historical_source_paired_dataset1_testA/`; the preserved reference
images in `results/historical_dataset1_testA/` are not overwritten.
