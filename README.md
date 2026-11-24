# ReFuGe: Feature Generation for Prediction Tasks on Relational Databases with LLM Agents

This is the official implementation of ReFuGe **(Feature Generation for Prediction Tasks on Relational Databases with LLM Agents)**.

Accepted in **ACM WWW 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2601.17735-b31b1b.svg)](https://arxiv.org/abs/2601.17735)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

### 🛠️ Requirements

ReFuGe requires core dependencies:

RelBench library (required) — used for loading RDB benchmark datasets, schemas, and task definitions.

Install RelBench via:

```bash
pip install relbench
```
For details, please refer to https://github.com/snap-stanford/relbench

Claude API — ReFuGe uses Claude models as the backbone LLM agents
(schema selection, feature generation, and feature filtering).
Make sure to set your API key:

```bash
export ANTHROPIC_API_KEY="YOUR_API_KEY"
```

### 🚀 How to run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="YOUR_API_KEY"

python run.py --task f1-dnf
```

Available tasks: `f1-dnf`, `f1-top3`, `event-repeat`, `event-ignore`,
`avito-clicks`, `avito-visits`, `amazon-churn`.

Useful options:

| option | default | description |
|---|---|---|
| `--n_iter` | 3 | max iterations (stops early if an iteration keeps no feature) |
| `--models` | per task | 3 comma-separated planner model IDs |
| `--judge_model` | `claude-sonnet-4-20250514` | selects 3 of 9 plans and writes the feature code |
| `--temp` | 0.2 | base temperature (planner *k* uses `temp + 0.2k`) |
| `--delta` | 0.0 | keep a feature only if val AUROC ≥ best + delta |
| `--out_dir` | `outputs/<dataset>/<task>` | where results are written |
| `--seed` | none | seed for the evaluator's train subsampling |

Outputs (`--out_dir`):
```
<task>-{train,val,test}.csv   task tables augmented with all accepted features
features/                     per-feature CSVs produced by the generated code
generated_code/iter<i>.py     LLM-written feature code for each iteration
```

### 📁 Code structure

```
run.py              CLI entry point
refuge/
  tasks.py          per-task settings (entity table/key, target, evaluator columns, models)
  pipeline.py       main loop: table → column selection → 9 plans → judge → code → validation filter
  prompts.py        prompt templates for every step
  evaluate.py       LightGBM validation-AUROC check
  parsing.py        parsing of plans, judge JSON and generated code
  llm.py            Anthropic API wrapper
data/rel-*/         schema, column statistics and task description given to the LLM
```

To add a task, add a `TaskConfig` entry in `refuge/tasks.py` (schema/statistics/task
files for rel-hm, rel-stack and rel-trial are already under `data/`).
