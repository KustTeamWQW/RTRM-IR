# Dataset1 V77 Metrics

Metric implementation:

```text
metric_reference/IQA我自己的/IQA
```

Evaluation follows `IQA我自己的/IQA/test.py`:

- `cv2.imread`
- sorted filename pairing
- `crop_border=4` for PSNR and SSIM
- grayscale result image for AG, EME, and NIQE

## Preserved Historical Output

Result directory:

```text
results/historical_dataset1_testA/
```

| Metric | Direction | Mean |
|---|---:|---:|
| PSNR | ↑ | 32.485579151761236 |
| SSIM | ↑ | 0.83698733726667 |
| AG | ↑ | 3.2881651675561683 |
| EME | ↑ | 2.458763262407466 |
| NIQE | ↓ | 18.918707629932836 |

Detailed files:

- `user_iqa_historical_saved_outputs_all_metrics.json`
- `user_iqa_historical_saved_outputs_all_metrics.csv`
- `user_iqa_historical_saved_outputs_all_metrics.log`

## Historical Source Reproduction

Result directory:

```text
results/historical_source_paired_dataset1_testA/
```

The output was generated from `historical_v77_source/` and the bundled V77
checkpoint. All 160 PNG files are byte identical to
`results/historical_dataset1_testA/`. The bundled IQA implementation reports:

| Metric | Direction | Mean |
|---|---:|---:|
| PSNR | ↑ | 32.485579151761236 |
| SSIM | ↑ | 0.83698733726667 |
| AG | ↑ | 3.2881651675561683 |
| EME | ↑ | 2.458763262407466 |
| NIQE | ↓ | 18.918707629932836 |

Detailed files:

- `user_iqa_historical_source_paired_fullsize.json`
- `user_iqa_historical_source_paired_fullsize.csv`
