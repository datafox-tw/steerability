"""Print a shareable Week 1 environment report without modifying the environment."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

import torch

from steerability.algorithms.core.registry import REGISTRY


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    cuda_available = torch.cuda.is_available()
    report = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "packages": {
            name: package_version(name)
            for name in ("steerability", "torch", "transformers", "datasets", "inspect-ai", "fsspec")
        },
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count(),
        "visible_gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_vram_gib": round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 1),
            }
            for index in range(torch.cuda.device_count())
        ]
        if cuda_available
        else [],
        "registry": {category: sorted(methods) for category, methods in REGISTRY.items()},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
