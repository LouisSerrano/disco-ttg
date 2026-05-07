"""Upload DISCO model checkpoints to a HuggingFace model repo.

Run *after* `huggingface-cli login` (or set HF_TOKEN env var).

Default behaviour: pushes the four paper-final DISCO checkpoints listed below
to sogeeking/disco-models, under per-equation subfolders.

Usage:
    python scripts/upload_checkpoints_to_hf.py
    python scripts/upload_checkpoints_to_hf.py --repo_id myorg/disco --dry_run
"""
import argparse
import os

from huggingface_hub import HfApi, create_repo, upload_file


# Paper checkpoints (verified to exist 2026-05).
PAPER_CKPTS = {
    "advection-diffusion": {
        "src": (
            "./outputs/"
            "DISCO_advection-diffusion_solverrk4_adjFalse_h128_t2_steps1_initTrue"
            "_bs512_lr0.001_ctxTrue_noise0_inframes16_outframes16_T10/last.ckpt"
        ),
        "hf_path": "advection-diffusion/last.ckpt",
    },
    "navier-stokes": {
        "src": (
            "./outputs/"
            "DISCO_euler_solverrk4_adjFalse_h128_t4_steps4_initFalse_bs16_lr0.0003"
            "_hdf5_noise0_inframes16_outframes2_subx1_subt1_20260124_041037/best-checkpoint.ckpt"
        ),
        "hf_path": "navier-stokes/best-checkpoint.ckpt",
    },
    "combined-equation": {
        "src": (
            "./outputs/"
            "DISCO_combined-physics-hdf5_solverrk4_adjFalse_h128_t3_steps1_initTrue"
            "_bs64_lr0.0005_hdf5_noise0_inframes16_outframes16_subx1_subt1_20250902_141155/last.ckpt"
        ),
        "hf_path": "combined-equation/last.ckpt",
    },
    # Reaction-diffusion: pick the latest one matching the paper config.
    "reaction-diffusion": {
        "src": (
            "./outputs/"
            "DISCO_rd_solverrk4_adjFalse_h128_t3_steps1_initFalse_bs64_lr0.0003"
            "_hdf5_noise0_inframes16_outframes2_subx1_subt1_20250916_204203/last.ckpt"
        ),
        "hf_path": "reaction-diffusion/last.ckpt",
    },
}


README_TEMPLATE = """---
license: mit
tags:
  - pytorch-lightning
  - neural-pde-solver
  - operator-splitting
---

# DISCO model checkpoints

Pretrained checkpoints for the ICML 2026 paper *Test-Time Generalization via Neural
Operator Splitting* (Serrano et al.).

| Equation | Checkpoint |
|---|---|
| Advection-diffusion (1D) | `advection-diffusion/last.ckpt` |
| Combined equation (1D) | `combined-equation/last.ckpt` |
| Reaction-diffusion (Gray-Scott, 2D) | `reaction-diffusion/last.ckpt` |
| Navier-Stokes / Euler (2D) | `navier-stokes/best-checkpoint.ckpt` |

Load with PyTorch Lightning:

```python
from train.train_generic import DISCOLitModule
model = DISCOLitModule.load_from_checkpoint("last.ckpt")
```

Code: https://github.com/LouisSerrano/disco-ttg
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="sogeeking/disco-models")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                   help="Print actions without uploading")
    p.add_argument("--equations", nargs="+", default=list(PAPER_CKPTS.keys()),
                   help="Subset of equations to upload")
    args = p.parse_args()

    api = HfApi()

    print(f"Repo: {args.repo_id} | private={args.private}")
    if not args.dry_run:
        create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        # Push README
        with open("/tmp/_disco_models_README.md", "w") as fh:
            fh.write(README_TEMPLATE)
        upload_file(
            path_or_fileobj="/tmp/_disco_models_README.md",
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
        )

    for eq in args.equations:
        meta = PAPER_CKPTS[eq]
        src = meta["src"]
        if not os.path.exists(src):
            print(f"  SKIP {eq}: source not found ({src})")
            continue
        size_mb = os.path.getsize(src) / (1024 * 1024)
        print(f"  {eq}  {src}  ({size_mb:.1f} MB)  ->  {args.repo_id}/{meta['hf_path']}")
        if args.dry_run:
            continue
        upload_file(
            path_or_fileobj=src,
            path_in_repo=meta["hf_path"],
            repo_id=args.repo_id,
            repo_type="model",
        )

    print("Done." if not args.dry_run else "(dry-run; nothing uploaded)")


if __name__ == "__main__":
    main()
