# CoCoBench: Multi-Agent Embodied Coordination Benchmark

This repository contains an AI2-THOR/iTHOR benchmark for evaluating
multi-agent high-level planning. The benchmark covers task allocation,
temporal dependencies, resource conflicts, and relay handoffs with two to four
agents.

The repository contains source code and task definitions only. Generated
trajectories, images, logs, evaluation reports, and model comparison results are
intentionally excluded.

## Repository layout

```text
benchmark/
  configs/       Evaluation configuration templates
  eval/          Oracle, policy, runner, and metric implementations
  task_config/   Benchmark instances and evaluation-set index
  tools/         Instance generation and validation utilities
  *.py           Environment, actions, navigation, and skill execution
requirements.txt
```

## Installation

Python 3.9 or later is recommended. AI2-THOR rendering may additionally require
a working display or Vulkan-capable GPU, depending on the selected platform.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd benchmark
```

## Oracle smoke test

Run one task with its built-in reference plan:

```bash
python eval/runner.py \
  --task-config task_config/A/D1/A_D1__FloorPlan1__seed0.json \
  --oracle-plan auto
```

Generated files are written under `benchmark/outputs/`, which is ignored by
Git.

## Offline policy smoke test

The mock backend exercises the evaluation pipeline without network access or an
API key:

```bash
python eval/evaluate_vlm.py \
  --vlm-backend mock \
  --eval-set rep240 \
  --limit 2 \
  --max-steps 3 \
  --exp-name mock-smoke
```

## VLM evaluation

Copy `benchmark/configs/config.yaml` to a local configuration if you need to
change the endpoint or defaults. Keep credentials in environment variables:

```bash
export OPENAI_API_KEY="<your-api-key>"
export OPENAI_BASE_URL="https://api.openai.com/v1"

python eval/evaluate_vlm.py \
  --config config.yaml \
  --model-name <model-id> \
  --eval-set rep240 \
  --exp-name example-run
```

For an Anthropic-compatible endpoint, select `--vlm-backend anthropic` and set
`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, and `ANTHROPIC_MODEL`.

The batch helper can run centralized and distributed conditions:

```bash
bash run_eval.sh \
  --model <model-id> \
  --conditions C0_image,D0_image,D1_image \
  --eval-set rep240 \
  --jobs 4
```

## Dataset

`benchmark/task_config/index.json` indexes 897 instances across four
coordination dimensions:

- `D1`: independent parallelism and task allocation
- `D2`: sequential dependencies and temporal planning
- `D3`: resource exclusion and conflict resolution
- `D4`: relay handoff and producer-consumer coordination

The `rep240` curated subset can be used for smaller evaluation runs. Use
`tools/generate_index.py` after changing task definitions.

## Validation and metrics

```bash
# Validate a subset with oracle policies.
python tools/validate_instances.py --only-cell K_D3 --gpu-device 0

# Compute metrics for a generated trajectory.
python eval/metrics.py outputs/task_runs/<task-id>/trajectory.json
```

Metrics are computed from trajectories and task predicates without a separate
judge model.
