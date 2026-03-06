"""
Print parameter counts for one or more nanochat checkpoints.

Examples:
    python -m scripts.model_size /path/to/base_checkpoints/d12
    python -m scripts.model_size /path/to/base_checkpoints/d12/model_002205.pt /path/to/other/model_dir
    python -m scripts.model_size /path/to/model_dir --step 2205
"""

import argparse
import os
import re

import torch

from nanochat.checkpoint_manager import build_model, find_last_step


MODEL_FILE_RE = re.compile(r"^model_(\d+)\.pt$")


def format_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.3f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3f}M"
    if n >= 1_000:
        return f"{n / 1_000:.3f}K"
    return str(n)


def resolve_checkpoint(path: str, step: int | None) -> tuple[str, int]:
    full_path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(full_path):
        match = MODEL_FILE_RE.match(os.path.basename(full_path))
        if not match:
            raise ValueError(f"Expected file name like model_000123.pt, got: {full_path}")
        return os.path.dirname(full_path), int(match.group(1))

    if os.path.isdir(full_path):
        resolved_step = find_last_step(full_path) if step is None else step
        expected_model_file = os.path.join(full_path, f"model_{resolved_step:06d}.pt")
        if not os.path.exists(expected_model_file):
            raise FileNotFoundError(f"Missing checkpoint file: {expected_model_file}")
        return full_path, resolved_step

    raise FileNotFoundError(f"Path does not exist: {full_path}")


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser(description="Print model sizes (#params) for checkpoint paths")
    parser.add_argument(
        "model_paths",
        type=str,
        nargs="+",
        help="List of checkpoint dirs or model_XXXXXX.pt files",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step to load when a model path is a directory (default: latest)",
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to load checkpoints on (default: cpu)",
    )
    args = parser.parse_args()

    if args.device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if args.device_type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")

    device = torch.device(args.device_type)
    print("model_path,step,total_params,total_params_human,trainable_params,trainable_params_human")

    for model_path in args.model_paths:
        try:
            checkpoint_dir, step = resolve_checkpoint(model_path, args.step)
            model, _, _ = build_model(checkpoint_dir, step, device=device, phase="eval")
            total_params, trainable_params = count_params(model)
            print(
                f"{checkpoint_dir},{step},{total_params},{format_params(total_params)},"
                f"{trainable_params},{format_params(trainable_params)}"
            )
            del model
        except Exception as exc:
            print(f"{model_path},ERROR,{exc}")


if __name__ == "__main__":
    main()
