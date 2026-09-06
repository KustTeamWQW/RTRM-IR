# VIFPM

Infrared image enhancement experiments focused on radiation-texture
decomposition and infrared-only inference.

## Active Code

The current cleaned implementation has been copied to the repository root:

```text
train.py
data/
models/
options/
util/
IQA/
```

Use this root-level launcher for the current from-scratch `datasets2` experiment:

```powershell
.\reproduce\launch_datasets2_v23_fromscratch.ps1
```

The launcher starts the v23 pruned-loss version:

```text
datasets2_texture_residual_gtvis_v23_prunedloss_ir_noclahe_vis_clahe_10ep_seed20260728
```

## Reference Code

`checkpoint-data1/` is kept as the original reference implementation and should
not be modified when changing the experimental version.

The previous experimental working copy is still kept under
`experiments/datasets2_texture_fromscratch/` for traceability.

## Train/Test Boundary

Visible images are used only during training as texture reference supervision.
Testing and deployment use infrared inputs only.

## GitHub Upload

Large generated files are ignored by `.gitignore`, including datasets,
checkpoints, logs, preprocessing caches, and Python cache files.

## Environment

The training scripts are configured for the local `py3.7` environment at:

```text
D:\Environment_2023\Anaconda3\envs\py3.7
```

Package versions exported from that environment are stored in:

```text
requirements-py37.txt
environment-py37.yml
```
