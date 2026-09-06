# RTRM-IR: RGB-Texture-Guided Infrared Image Enhancement with Thermal Radiance Modeling

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
