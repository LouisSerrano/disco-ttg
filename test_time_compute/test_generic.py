"""Generic test-time compute evaluation for DISCO.

Loads a checkpoint trained with train/train_generic.py, builds a dictionary of
operators (from the codebook OR by encoding training trajectories), then runs
a chosen operator-selection strategy on a held-out HDF5 (or HuggingFace-Hub)
dataset.

Two ways to build the operator dictionary:
    --operator_source codebook  : take rows from the trained codebook directly
    --operator_source encoder   : encode trajectories from the train set with
                                  the encoder and average within each environment

Selection strategies (--method): direct | random | greedy | beam

Example:
    python test_time_compute/test_generic.py \
        --model_path /path/to/best-checkpoint.ckpt \
        --train_files /path/to/train.h5 \
        --test_files  /path/to/test.h5 \
        --operator_source encoder \
        --method beam --beam_width 3
"""
import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from test_time_compute.ttc_utils import DEVICE
from test_time_compute.ttc_methods import (
    test_direct_prediction,
    random_operator_selection_batch,
    greedy_operator_selection,
    beam_search_operator_selection_batch,
)
from train.train_generic import DISCOLitModule, GenericHDF5Dataset, _resolve_files


def load_model_from_checkpoint(checkpoint_path):
    """Load DISCO model + Lightning module from a generic-train checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    lit = DISCOLitModule.load_from_checkpoint(checkpoint_path, map_location=DEVICE)
    lit.eval()
    model = lit.model.to(DEVICE).eval()
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} params")
    return model, lit


def build_operator_dict_from_codebook(lit_model, n_operators=None, only_used=True):
    """Take operators directly from the trained codebook.

    Args:
        lit_model: the loaded DISCOLitModule
        n_operators: if set, take the top-n by usage; else return all
        only_used: drop entries with zero EMA usage (untrained envs)
    Returns:
        Tensor (n, theta_dim) — pre-decode latent.
    """
    codebook = lit_model.codebook.detach().cpu()
    usage = lit_model.codebook_usage.detach().cpu()

    if only_used:
        keep = usage > 0
        codebook = codebook[keep]
        usage = usage[keep]
        if codebook.numel() == 0:
            raise ValueError("Codebook has no used entries — was the model trained?")

    if n_operators is not None and codebook.shape[0] > n_operators:
        topk = torch.topk(usage, k=n_operators).indices
        codebook = codebook[topk]

    print(f"Codebook operators: {codebook.shape[0]} (dim {codebook.shape[1]})")
    return codebook


def build_operator_dict_from_encoder(model, dataloader, n_operators, n_per_op=4):
    """Encode trajectories with the encoder, average per environment.

    Returns a tensor (n_operators, theta_dim) — pre-decode latent.
    """
    print(f"Encoding {n_operators} operators (×{n_per_op} trajectories each)...")
    by_env = {}
    state_labels = torch.tensor([0], device=DEVICE)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inp = batch["input"].to(DEVICE)
            envs = batch["environment_idx"].cpu().tolist()
            theta, _ = model.encode_theta_latent(inp, state_labels)
            theta = theta.cpu()
            for i, e in enumerate(envs):
                by_env.setdefault(int(e), []).append(theta[i])
            # stop once we have enough envs all with enough samples
            full = sum(1 for v in by_env.values() if len(v) >= n_per_op)
            if full >= n_operators:
                break

    keys = sorted(k for k, v in by_env.items() if len(v) >= n_per_op)[:n_operators]
    if len(keys) < n_operators:
        print(f"  warning: only collected {len(keys)} envs with ≥{n_per_op} samples")
    ops = torch.stack([torch.stack(by_env[k][:n_per_op], 0).mean(0) for k in keys], 0)
    print(f"Built encoder dictionary: {ops.shape[0]} operators (dim {ops.shape[1]})")
    return ops


def _run_selection_per_sample(method_fn, model, theta_latent_operators, test_loader,
                               max_samples, dt, **kwargs):
    """Iterate test_loader, run a per-sample selection method, collect errors."""
    errors = []
    sample_idx = 0
    t0 = time.time()
    for batch in test_loader:
        inp = batch["input"].to(DEVICE)
        tgt = batch["target"].to(DEVICE)
        for i in range(inp.size(0)):
            if sample_idx >= max_samples:
                break
            composition, error, _pred = method_fn(
                model, theta_latent_operators,
                inp[i : i + 1], tgt[i : i + 1],
                dt=dt, **kwargs,
            )
            errors.append(float(error))
            if sample_idx % 50 == 0:
                print(f"  sample {sample_idx}: composition={composition} err={error:.4f}")
            sample_idx += 1
        if sample_idx >= max_samples:
            break
    print(f"Processed {len(errors)} samples in {time.time() - t0:.1f}s")
    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--train_files", nargs="+", default=None,
                   help="HDF5 train files (only used when --operator_source encoder)")
    p.add_argument("--test_files", nargs="+", required=True, help="HDF5 test files")
    p.add_argument("--hf_repo_id", type=str, default=None,
                   help="If set, download files from this HF dataset repo first")
    p.add_argument("--trajectories_key", type=str, default="trajectories")
    p.add_argument("--env_id_key", type=str, default="env_id")
    p.add_argument("--n_input_frames", type=int, default=16)
    p.add_argument("--n_output_frames", type=int, default=16)
    p.add_argument("--operator_source", choices=["codebook", "encoder"], default="codebook")
    p.add_argument("--n_operators", type=int, default=64)
    p.add_argument("--n_per_op", type=int, default=4)
    p.add_argument("--method", choices=["direct", "random", "greedy", "beam"], default="random")
    p.add_argument("--max_operators", type=int, default=5,
                   help="Max operators in a composition (greedy/beam)")
    p.add_argument("--random_trials", type=int, default=64)
    p.add_argument("--random_batch_size", type=int, default=16)
    p.add_argument("--composition_lengths", type=int, nargs="+", default=[2, 3])
    p.add_argument("--beam_width", type=int, default=3)
    p.add_argument("--beam_batch_size", type=int, default=32)
    p.add_argument("--splitting_method", choices=["strang", "lie"], default="strang")
    p.add_argument("--dt", type=float, default=10.0 / 100,
                   help="Time step between consecutive frames in the trajectory")
    p.add_argument("--num_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output_dir", type=str, default="./results/generic_ttc")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run timestamp: {timestamp}")

    # ---------- Resolve files (HF or local) ----------
    class _DataCfg:
        hf_repo_id = args.hf_repo_id
        hf_revision = None
        train_files = args.train_files or []
        test_files = args.test_files

    test_files = _resolve_files(_DataCfg, "test")
    print(f"Test files: {test_files}")

    test_ds = GenericHDF5Dataset(
        test_files,
        n_input_frames=args.n_input_frames,
        n_output_frames=args.n_output_frames,
        trajectories_key=args.trajectories_key,
        env_id_key=args.env_id_key,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Test set: {len(test_ds)} samples")

    # ---------- Load model ----------
    model, lit = load_model_from_checkpoint(args.model_path)

    # ---------- Build operator dictionary ----------
    if args.operator_source == "codebook":
        ops_latent = build_operator_dict_from_codebook(lit, n_operators=args.n_operators)
    else:
        if not args.train_files:
            raise ValueError("--train_files is required when --operator_source encoder")
        train_files = _resolve_files(_DataCfg, "train")
        train_ds = GenericHDF5Dataset(
            train_files,
            n_input_frames=args.n_input_frames,
            n_output_frames=args.n_output_frames,
            trajectories_key=args.trajectories_key,
            env_id_key=args.env_id_key,
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2
        )
        ops_latent = build_operator_dict_from_encoder(
            model, train_loader, n_operators=args.n_operators, n_per_op=args.n_per_op
        )

    # ---------- Run the selected strategy ----------
    print(f"\nMethod: {args.method}  |  splitting: {args.splitting_method}  |  dt: {args.dt}")
    if args.method == "direct":
        avg_err, _preds = test_direct_prediction(model, test_loader, dt=args.dt)
        errors = [float(avg_err)]
        summary = {"avg_error": float(avg_err)}
    elif args.method == "random":
        errors = _run_selection_per_sample(
            random_operator_selection_batch, model, ops_latent, test_loader,
            args.num_samples, args.dt,
            num_compositions=args.random_trials,
            composition_lengths=args.composition_lengths,
            random_batch_size=args.random_batch_size,
            splitting_method=args.splitting_method,
        )
        summary = {"avg_error": float(np.mean(errors))}
    elif args.method == "greedy":
        errors = _run_selection_per_sample(
            greedy_operator_selection, model, ops_latent, test_loader,
            args.num_samples, args.dt,
            max_operators=args.max_operators,
            splitting_method=args.splitting_method,
        )
        summary = {"avg_error": float(np.mean(errors))}
    elif args.method == "beam":
        errors = _run_selection_per_sample(
            beam_search_operator_selection_batch, model, ops_latent, test_loader,
            args.num_samples, args.dt,
            beam_width=args.beam_width,
            max_operators=args.max_operators,
            splitting_method=args.splitting_method,
        )
        summary = {"avg_error": float(np.mean(errors))}

    # ---------- Save ----------
    out_path = os.path.join(args.output_dir, f"{args.method}_{args.operator_source}_{timestamp}.npz")
    np.savez(out_path, errors=np.array(errors), args=json.dumps(vars(args)))
    print(f"\nFinal mean rel-L2: {summary['avg_error']:.4f}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
