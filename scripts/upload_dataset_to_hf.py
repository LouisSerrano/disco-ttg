"""Upload a single equation's HDF5 dataset to a HuggingFace dataset repo.

Designed for files in the *generic* format produced by
scripts/convert_to_generic_hdf5.py (i.e. with `trajectories` + `env_id` keys).

Run *after* `huggingface-cli login`.

Usage:
    python scripts/upload_dataset_to_hf.py \
        --repo_id sogeeking/disco-ad \
        --files /data/E_AD_train.h5 /data/E_AD_val.h5 /data/E_AD_test.h5
"""
import argparse
import os

from huggingface_hub import create_repo, upload_file


README_TEMPLATE = """---
license: cc-by-4.0
tags:
  - physics
  - pde
  - neural-operator
---

# {name}

Trajectories used in the ICML 2026 paper *Test-Time Generalization via Neural
Operator Splitting* (Serrano et al.).

## Format

Each HDF5 file contains:
- `trajectories`: shape `(N, T, C, *spatial)`, float32
- `env_id`:       shape `(N,)`, int64 — environment index for each trajectory
- `env_params/*`: optional metadata mapping env_id back to PDE coefficients

Load via the project's `train_generic.py` / `test_generic.py` scripts:

```python
from train.train_generic import GenericHDF5Dataset
ds = GenericHDF5Dataset(["train.h5"])
```

Or using `huggingface_hub`:

```python
from huggingface_hub import hf_hub_download
local = hf_hub_download(repo_id="{repo_id}", filename="train.h5", repo_type="dataset")
```

Code: https://github.com/LouisSerrano/neural-operator-splitting
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", required=True, help="e.g. sogeeking/disco-ad")
    p.add_argument("--files", nargs="+", required=True, help="HDF5 files to upload")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    name = args.repo_id.split("/", 1)[-1]
    print(f"Repo: {args.repo_id} | private={args.private}")

    if not args.dry_run:
        create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
        readme_path = "/tmp/_disco_dataset_README.md"
        with open(readme_path, "w") as fh:
            fh.write(README_TEMPLATE.format(name=name, repo_id=args.repo_id))
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    for src in args.files:
        if not os.path.exists(src):
            print(f"  SKIP: {src} not found")
            continue
        size_mb = os.path.getsize(src) / (1024 * 1024)
        target = os.path.basename(src)
        print(f"  {src}  ({size_mb:.1f} MB)  ->  {args.repo_id}/{target}")
        if args.dry_run:
            continue
        upload_file(
            path_or_fileobj=src,
            path_in_repo=target,
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    print("Done." if not args.dry_run else "(dry-run; nothing uploaded)")


if __name__ == "__main__":
    main()
