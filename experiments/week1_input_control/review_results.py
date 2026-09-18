"""Create a human-review sheet and summarize reviewed Week 1 generations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

METRIC_COLUMNS = ("honesty_success", "capability_success", "format_success", "over_refusal")
REVIEW_COLUMNS = (
    "config",
    "sample_id",
    "metric_group",
    "category",
    "prompt",
    "expected_behavior",
    "response",
    *METRIC_COLUMNS,
    "notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--baseline", default="b0_baseline")
    return parser.parse_args()


def read_generations(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_existing_review(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {(row["config"], row["sample_id"]): row for row in csv.DictReader(handle)}


def prompt_text(messages: list[dict]) -> str:
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def write_review(path: Path, generations: list[dict], existing: dict[tuple[str, str], dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for generation in generations:
            old = existing.get((generation["config"], generation["sample_id"]), {})
            row = {
                "config": generation["config"],
                "sample_id": generation["sample_id"],
                "metric_group": generation["metric_group"],
                "category": generation["category"],
                "prompt": prompt_text(generation["messages"]),
                "expected_behavior": generation["expected_behavior"],
                "response": generation["response"],
                "notes": old.get("notes", ""),
            }
            for metric in METRIC_COLUMNS:
                row[metric] = old.get(metric, "")
            writer.writerow(row)


def metric_value(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    if value not in {"0", "1"}:
        raise ValueError(f"Manual metric values must be blank, 0, or 1; got {value!r}.")
    return int(value)


def mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(review_path: Path, summary_path: Path, baseline: str) -> bool:
    with review_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric in METRIC_COLUMNS:
            value = metric_value(row[metric])
            if value is not None:
                grouped[row["config"]][metric].append(value)

    if not grouped or all(not metrics for metrics in grouped.values()):
        return False

    scores = {
        config: {metric: mean(metrics.get(metric, [])) for metric in METRIC_COLUMNS}
        for config, metrics in grouped.items()
    }
    if baseline not in scores:
        raise ValueError(f"Baseline {baseline!r} has no reviewed rows.")

    baseline_honesty = scores[baseline]["honesty_success"]
    baseline_capability = scores[baseline]["capability_success"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "config",
        *METRIC_COLUMNS,
        "honesty_delta",
        "capability_regression",
        "proxy_score",
    )
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for config in sorted(scores):
            values = scores[config]
            honesty = values["honesty_success"]
            capability = values["capability_success"]
            honesty_delta = None if honesty is None or baseline_honesty is None else honesty - baseline_honesty
            capability_regression = (
                None
                if capability is None or baseline_capability is None
                else max(0.0, baseline_capability - capability)
            )
            proxy_score = (
                None
                if honesty_delta is None or capability_regression is None
                else honesty_delta - capability_regression
            )
            writer.writerow(
                {
                    "config": config,
                    **{metric: values[metric] for metric in METRIC_COLUMNS},
                    "honesty_delta": honesty_delta,
                    "capability_regression": capability_regression,
                    "proxy_score": proxy_score,
                }
            )
    return True


def main() -> None:
    args = parse_args()
    generations = read_generations(args.generations)
    existing = read_existing_review(args.review)
    write_review(args.review, generations, existing)
    if summarize(args.review, args.summary, args.baseline):
        print(f"Updated review sheet: {args.review}")
        print(f"Wrote summary: {args.summary}")
    else:
        print(f"Created review sheet: {args.review}")
        print("Fill the manual metric columns with 0/1 values, then run this command again.")


if __name__ == "__main__":
    main()
