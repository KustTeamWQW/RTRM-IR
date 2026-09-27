#!/usr/bin/env python
"""Evaluate saved outputs with the metric implementation bundled in metrics/.

The evaluation follows metrics/test.py:
  - cv2.imread(...), default BGR/color loading
  - sorted image lists
  - crop_border=4
  - metrics/psnr_ssim.py for PSNR and SSIM
  - grayscale conversion for AG, EME, and NIQE
"""

import argparse
import csv
import json
import os
import sys
import typing
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


# torchvision 0.14 is the pinned version, but this keeps the Python 3.7
# runner compatible if a newer torchvision build is already installed.
if not hasattr(typing, 'OrderedDict'):
    typing.OrderedDict = OrderedDict


def _load_user_iqa(iqa_root):
    iqa_root = os.path.abspath(iqa_root)
    if iqa_root not in sys.path:
        sys.path.insert(0, iqa_root)
    from AG import avgGradient
    from EME import eme
    from niqe_1 import niqe_1
    from psnr_ssim import calculate_psnr, calculate_ssim
    return calculate_psnr, calculate_ssim, avgGradient, eme, niqe_1


def _image_files(directory):
    suffixes = {'.png', '.jpg', '.jpeg', '.bmp'}
    return sorted(
        path for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def evaluate(
    reference_dir,
    result_dir,
    calculate_psnr,
    calculate_ssim,
    avg_gradient,
    eme,
    niqe,
    crop_border,
):
    reference_files = _image_files(reference_dir)
    result_files = _image_files(result_dir)
    count = min(len(reference_files), len(result_files))
    psnr_values = []
    ssim_values = []
    ag_values = []
    eme_values = []
    niqe_values = []
    records = []

    for ref_path, result_path in zip(reference_files[:count], result_files[:count]):
        ref = cv2.imread(str(ref_path))
        result = cv2.imread(str(result_path))
        if ref is None or result is None:
            continue
        if ref.shape != result.shape:
            raise ValueError(
                f'Shape mismatch: {ref_path.name} {ref.shape} vs '
                f'{result_path.name} {result.shape}'
            )
        psnr = float(calculate_psnr(
            ref, result, crop_border=crop_border,
            input_order='HWC', test_y_channel=False
        ))
        ssim = float(calculate_ssim(
            ref, result, crop_border=crop_border,
            input_order='HWC', test_y_channel=False
        ))
        gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        ag = float(avg_gradient(gray_result))
        eme_score = float(eme(gray_result))
        niqe_score = float(niqe(gray_result))
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        ag_values.append(ag)
        eme_values.append(eme_score)
        niqe_values.append(niqe_score)
        records.append({
            'reference': str(ref_path),
            'result': str(result_path),
            'psnr': psnr,
            'ssim': ssim,
            'ag': ag,
            'eme': eme_score,
            'niqe': niqe_score,
        })

    if not records:
        raise RuntimeError('No valid image pairs were evaluated.')

    return {
        'reference_dir': str(reference_dir),
        'result_dir': str(result_dir),
        'reference_images': len(reference_files),
        'result_images': len(result_files),
        'evaluated_images': len(records),
        'pairing': 'sorted filename list, matching metrics/test.py',
        'crop_border': int(crop_border),
        'psnr': float(np.mean(psnr_values)),
        'ssim': float(np.mean(ssim_values)),
        'ag': float(np.mean(ag_values)),
        'eme': float(np.mean(eme_values)),
        'niqe': float(np.mean(niqe_values)),
        'score_psnr_plus_30_ssim': float(np.mean(psnr_values) + 30.0 * np.mean(ssim_values)),
        'records': records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iqa_root', default='')
    parser.add_argument('--reference_dir', required=True)
    parser.add_argument('--result_dir', required=True)
    parser.add_argument('--label', default='dataset1_odinmj')
    parser.add_argument('--crop_border', type=int, default=4)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--output_csv', required=True)
    args = parser.parse_args()

    bundle_dir = Path(__file__).resolve().parent
    iqa_root = args.iqa_root or str(bundle_dir / 'metrics')
    calculate_psnr, calculate_ssim, avg_gradient, eme, niqe = _load_user_iqa(iqa_root)
    result = evaluate(
        Path(args.reference_dir),
        Path(args.result_dir),
        calculate_psnr,
        calculate_ssim,
        avg_gradient,
        eme,
        niqe,
        args.crop_border,
    )
    result['label'] = args.label
    result['metric_implementation'] = iqa_root

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['label', 'psnr', 'ssim', 'ag', 'eme', 'niqe'],
        )
        writer.writeheader()
        writer.writerow({
            'label': args.label,
            'psnr': result['psnr'],
            'ssim': result['ssim'],
            'ag': result['ag'],
            'eme': result['eme'],
            'niqe': result['niqe'],
        })

    print(json.dumps({
        key: value for key, value in result.items() if key != 'records'
    }, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
