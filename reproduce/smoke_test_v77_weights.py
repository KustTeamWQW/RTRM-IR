import json
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
CKPT_NAME = "datasets1_V77_guidedK19_TirBlur31_cosPrompt_fusionRes1_scratch_bs4_ep520_gpu6_20260809_054006"
CKPT_DIR = ROOT / "checkpoints" / CKPT_NAME


def main():
    print("python smoke test")
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())

    with (CKPT_DIR / "train_opt.json").open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    cfg.update(
        {
            "checkpoints_dir": str(ROOT / "checkpoints"),
            "name": CKPT_NAME,
            "isTrain": True,
            "continue_train": True,
            "epoch": "best_val",
            "gpu_ids": [],
            "verbose": False,
        }
    )

    from models import create_model

    model = create_model(SimpleNamespace(**cfg))
    model.setup(model.opt)
    model.eval()

    x = torch.rand(1, 1, 256, 256)
    batch = {
        "infrared": x,
        "infrared-radiation": x.clone(),
        "gt-radiation": x.clone(),
        "visible": x.clone(),
        "gt": x.clone(),
        "paths": ["dummy.png"],
        "J_paths": ["dummy.png"],
    }
    model.set_input(batch)
    with torch.no_grad():
        model.forward()

    print("SMOKE_TEST_OK")
    print("fused_I_shape:", tuple(model.fused_I.shape))
    print("fused_I_minmax: %.6f %.6f" % (float(model.fused_I.min()), float(model.fused_I.max())))


if __name__ == "__main__":
    main()
