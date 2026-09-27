# Dataset1 V77 Reference Metrics

The 160 preserved model outputs are stored in `reference/outputs/`. Metrics
were calculated with the implementation in `metrics/`, using sorted filename
pairing, `cv2.imread`, a four-pixel crop for PSNR/SSIM, and grayscale output
images for AG, EME, and NIQE.

| Metric | Direction | Mean |
|---|---:|---:|
| PSNR | higher | 32.485579151761236 |
| SSIM | higher | 0.836987337266670 |
| AG | higher | 3.2881651675561683 |
| EME | higher | 2.458763262407466 |
| NIQE | lower | 18.918707629932836 |

Detailed per-image values are retained in
`user_iqa_historical_saved_outputs_all_metrics.json` and the corresponding
CSV and log files.
