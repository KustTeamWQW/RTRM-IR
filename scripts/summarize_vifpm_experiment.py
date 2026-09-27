#!/usr/bin/env python3
import argparse
import json
import os


def _read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _fmt_metric(row):
    return (
        'epoch={epoch} psnr={psnr:.4f} ssim={ssim:.4f} '
        'score={score:.4f} images={images}'
    ).format(**row)


def main():
    parser = argparse.ArgumentParser(description='Summarize VIFPM validation metrics for one experiment.')
    parser.add_argument('--checkpoints_dir', default='/home/jpc/jpc/VIFPM/checkpoints')
    parser.add_argument('--name', required=True)
    parser.add_argument('--target_psnr', type=float, default=30.0)
    parser.add_argument('--target_ssim', type=float, default=0.85)
    args = parser.parse_args()

    exp_dir = os.path.join(args.checkpoints_dir, args.name)
    metrics_path = os.path.join(exp_dir, 'val_metrics.jsonl')
    meta_path = os.path.join(exp_dir, 'best_val_meta.json')
    rows = _read_jsonl(metrics_path)

    result = {
        'name': args.name,
        'metrics_path': metrics_path,
        'num_val_records': len(rows),
        'target_psnr': args.target_psnr,
        'target_ssim': args.target_ssim,
    }
    if rows:
        latest = rows[-1]
        best = max(rows, key=lambda row: float(row.get('score', 0.0)))
        result.update({
            'latest_val': latest,
            'best_val_by_score': best,
            'latest_psnr_gap': args.target_psnr - float(latest.get('psnr', 0.0)),
            'latest_ssim_gap': args.target_ssim - float(latest.get('ssim', 0.0)),
            'best_psnr_gap': args.target_psnr - float(best.get('psnr', 0.0)),
            'best_ssim_gap': args.target_ssim - float(best.get('ssim', 0.0)),
            'target_reached_by_best': (
                float(best.get('psnr', 0.0)) >= args.target_psnr
                and float(best.get('ssim', 0.0)) >= args.target_ssim
            ),
        })
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as handle:
            result['best_val_meta'] = json.load(handle)

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if rows:
        print('LATEST_VAL ' + _fmt_metric(rows[-1]))
        print('BEST_VAL ' + _fmt_metric(max(rows, key=lambda row: float(row.get('score', 0.0)))))
    else:
        print('NO_VAL_RECORDS')


if __name__ == '__main__':
    main()
