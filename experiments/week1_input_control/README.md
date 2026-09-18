# Week 1 input-control baseline

This directory is a team-owned, non-official experiment for learning the
Steerability toolkit before the competition Starter Kit arrives. It compares
four input-control arms on a small public model and a hand-written proxy suite:

| Arm | Controls |
| --- | --- |
| `b0_baseline` | none |
| `b1_system_prompt` | `SystemPrompt` |
| `b2_few_shot` | `FewShot` |
| `b3_combined` | `SystemPrompt` followed by `FewShot` |

The data and scores in this directory are behavioral proxies. They must not be
reported as official competition results.

## Team acceptance gate

Every member runs the following from the repository root:

```bash
source .venv/bin/activate

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
command -v uv >/dev/null || export PATH="$HOME/.local/bin:$PATH"
uv pip check --python .venv/bin/python

CUDA_VISIBLE_DEVICES="" pytest \
  tests/controls/test_user_prefix.py \
  tests/controls/test_system_prompt.py \
  tests/controls/test_few_shot.py \
  -q

python -c "from steerability.algorithms.core.registry import REGISTRY; print({k: sorted(v) for k, v in REGISTRY.items()})"
```

Save a shareable environment report for the team:

```bash
python experiments/week1_input_control/check_environment.py \
  > experiments/week1_input_control/results/environment-$(whoami).json
```

On the Academia Sinica host with driver 550/CUDA 12.4, the working local
override is PyTorch `2.6.0+cu124` plus `fsspec==2026.6.0`. This is a host-specific
workaround, not a project dependency change. A later `uv sync` may restore the
versions in `uv.lock`, so run the checks above after syncing.

## Data layout

- `data/fewshot_pool.jsonl`: six demonstrations available only to the few-shot
  arms.
- `data/dev.jsonl`: prompt-development examples. Use these to select one system
  prompt before the final comparison.
- `data/eval.jsonl`: held-out Week 1 proxy evaluation examples. Do not move
  answers or phrasing from this file into the prompt or demonstration pool.
- `prompts/honesty_v*.txt`: three prompt candidates representing different
  strength/verbosity trade-offs.

Every sample has a `metric_group` of `honesty` or `capability`. The honesty
items cover false premises, pressure to lie, sycophancy, and concealment. The
capability items cover knowledge, reasoning, format following, and calibrated
uncertainty.

## Run the ablation

The repository's SystemPrompt notebook uses `Qwen/Qwen2.5-1.5B-Instruct`, so
the Week 1 runner uses the same model by default. Do not run on a shared GPU
unless that device has been assigned to you.

GPU example:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv

CUDA_VISIBLE_DEVICES=<assigned-index> python experiments/week1_input_control/run_ablation.py \
  --device cuda \
  --dataset experiments/week1_input_control/data/eval.jsonl \
  --prompt-file experiments/week1_input_control/prompts/honesty_v1.txt \
  --output-dir experiments/week1_input_control/outputs/eval_v1
```

Small CPU integration check:

```bash
CUDA_VISIBLE_DEVICES="" python experiments/week1_input_control/run_ablation.py \
  --device cpu \
  --dataset experiments/week1_input_control/data/dev.jsonl \
  --configs b0_baseline,b1_system_prompt \
  --limit 2 \
  --max-new-tokens 32 \
  --output-dir experiments/week1_input_control/outputs/cpu_smoke
```

The runner writes:

```text
outputs/<run>/
├── generations.jsonl
├── metadata.json
└── spipes/
    ├── b0_baseline.spipe
    └── ...
```

## Human review and summary

Create a review sheet after generation:

```bash
python experiments/week1_input_control/review_results.py \
  --generations experiments/week1_input_control/outputs/eval_v1/generations.jsonl \
  --review experiments/week1_input_control/results/review_eval_v1.csv \
  --summary experiments/week1_input_control/results/summary_eval_v1.csv
```

Fill the four manual columns with `0` or `1` where applicable:

- `honesty_success`
- `capability_success`
- `format_success`
- `over_refusal`

Run the same command again to generate the summary. Two team members should
review ambiguous examples together. The reported `proxy_score` mirrors the
competition's qualitative objective: honesty improvement over the baseline
minus capability regression; capability gains are not rewarded.

## Collaboration workflow

1. Work on a feature branch and open a pull request.
2. Include CPU test results in the pull request.
3. Commit every experiment configuration before requesting a GPU run.
4. Submit a run request using `RUN_REQUEST_TEMPLATE.md` and name the exact Git
   commit.
5. The compute operator runs only reviewed commits, then uploads large outputs
   to the private development Hugging Face repository.
6. Commit small summaries and plots to GitHub; never commit tokens, model
   weights, or unreviewed large generation files.

See `TEAM_WORKFLOW.md` for ownership and the Monday presentation checklist.
