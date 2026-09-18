"""Run the Week 1 B0-B3 input-control ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.input_control.few_shot.control import FewShot
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIGS = ("b0_baseline", "b1_system_prompt", "b2_few_shot", "b3_combined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset", type=Path, default=EXPERIMENT_DIR / "data" / "eval.jsonl")
    parser.add_argument("--fewshot-pool", type=Path, default=EXPERIMENT_DIR / "data" / "fewshot_pool.jsonl")
    parser.add_argument("--prompt-file", type=Path, default=EXPERIMENT_DIR / "prompts" / "honesty_v1.txt")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "outputs" / "latest")
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--k-positive", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=EXPERIMENT_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False.")
    return requested


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32
    return getattr(torch, requested)


def controls_for(config: str, prompt: str, examples: list[dict], k_positive: int) -> list:
    def system_prompt() -> SystemPrompt:
        return SystemPrompt(text=prompt, mode="prepend")

    def few_shot() -> FewShot:
        return FewShot(
            positive_example_pool=[{"prompt": row["prompt"], "response": row["response"]} for row in examples],
            k_positive=min(k_positive, len(examples)),
            selector="random",
            system_mode="append",
        )

    if config == "b0_baseline":
        return []
    if config == "b1_system_prompt":
        return [system_prompt()]
    if config == "b2_few_shot":
        return [few_shot()]
    if config == "b3_combined":
        return [system_prompt(), few_shot()]
    raise ValueError(f"Unknown config {config!r}; choose from {DEFAULT_CONFIGS}.")


def main() -> None:
    args = parse_args()
    configs = tuple(value.strip() for value in args.configs.split(",") if value.strip())
    unknown = sorted(set(configs) - set(DEFAULT_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown configs: {unknown}; choose from {DEFAULT_CONFIGS}.")
    if args.k_positive < 1:
        raise ValueError("--k-positive must be at least 1.")

    samples = read_jsonl(args.dataset)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")
        samples = samples[: args.limit]
    examples = read_jsonl(args.fewshot_pool)
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    if not samples or not examples or not prompt:
        raise ValueError("Dataset, few-shot pool, and prompt must all be non-empty.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spipe_dir = args.output_dir / "spipes"
    spipe_dir.mkdir(exist_ok=True)
    generations_path = args.output_dir / "generations.jsonl"

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model": args.model,
        "model_revision": args.model_revision,
        "dataset": str(args.dataset),
        "dataset_sha256": sha256(args.dataset),
        "fewshot_pool": str(args.fewshot_pool),
        "fewshot_pool_sha256": sha256(args.fewshot_pool),
        "prompt_file": str(args.prompt_file),
        "prompt_sha256": sha256(args.prompt_file),
        "configs": configs,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "device": device,
        "dtype": str(dtype),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with generations_path.open("w", encoding="utf-8") as output_handle:
        for config in configs:
            controls = controls_for(config, prompt, examples, args.k_positive)
            pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=controls)
            pipeline.steer()
            for control in controls:
                selector = getattr(control, "_selector", None)
                reseed = getattr(selector, "reseed", None)
                if callable(reseed):
                    reseed(args.seed)
            pipeline.to_spipe(model_ref=args.model).save(spipe_dir / f"{config}.spipe")

            for sample in samples:
                response = pipeline.generate(
                    messages=sample["messages"],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
                row = {
                    "config": config,
                    "sample_id": sample["id"],
                    "metric_group": sample["metric_group"],
                    "category": sample["category"],
                    "messages": sample["messages"],
                    "expected_behavior": sample["expected_behavior"],
                    "response": response,
                }
                output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_handle.flush()
                print(f"[{config}] {sample['id']}: {response[:120]!r}")

    print(f"Wrote generations to {generations_path}")


if __name__ == "__main__":
    main()
