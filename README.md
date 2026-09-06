# RTRM-IR: RGB-Texture-Guided Infrared Image Enhancement with Thermal Radiance Modeling

This repository contains the official project page for **RTRM-IR: RGB-texture-guided infrared image enhancement with thermal radiance modeling**.

The runnable project code is provided in the `RTRM-IR/` directory. The training data is not included in this repository.

## Environment

The project was tested with the local conda environment `py3.7`:

```text
Python 3.7.1
PyTorch 1.13.1+cu117
TorchVision 0.14.1+cu117
CUDA 11.7 package build
```

The exported environment files are:

```text
RTRM-IR/environment-py37.yml
RTRM-IR/requirements-py37.txt
```

To create a compatible environment:

```bash
conda create -n py3.7 python=3.7.1 pip=22.3.1
conda activate py3.7
pip install -r RTRM-IR/requirements-py37.txt
```

On the original Windows training machine, the training launcher uses:

```text
D:\Environment_2023\Anaconda3\envs\py3.7\python.exe
```

If your conda environment is installed elsewhere, update `$envPath` in:

```text
RTRM-IR/reproduce/launch_datasets1_v77_from_checkpoint_params.ps1
```

## Data

The default data root is:

```text
RTRM-IR/datasets1
```

Expected directory layout:

```text
datasets1/
  trainA/
  trainB/
  trainGT/
  testA/
  testB/
  testGT/
```
