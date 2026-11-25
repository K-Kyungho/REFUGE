# ReFuGe: Feature Generation for Prediction Tasks on Relational Databases with LLM Agents

This is the official implementation of ReFuGe **(Feature Generation for Prediction Tasks on Relational Databases with LLM Agents)**.

Accepted in **ACM WWW 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2601.17735-b31b1b.svg)](https://arxiv.org/abs/2601.17735)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)


### 🛠️ Requirements


ReFuGe uses [RelBench](https://github.com/snap-stanford/relbench) to load benchmark datasets and task definitions. The default Claude backend uses `ANTHROPIC_API_KEY`

OpenAI models and OpenAI-compatible servers are also supported (see [Using other LLM backbones](#-using-other-llm-backbones)).

### 🚀 How to run

```bash
git clone https://github.com/K-Kyungho/REFUGE.git
cd REFUGE
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
| `--provider` | `anthropic` | LLM backend: `anthropic` or `openai` |
| `--base_url` | none | custom endpoint for an OpenAI-compatible server (e.g. vLLM) |
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

### 🔌 Using other LLM backbones

The paper's experiments use Claude, but the backbone is configurable. Every LLM call goes
through `ask_model` in `refuge/llm.py`, and `--provider` selects the backend.

**OpenAI models**

```bash
pip install openai
export OPENAI_API_KEY="YOUR_API_KEY"

python run.py --task f1-dnf --provider openai \
    --models gpt-4.1,gpt-4.1-mini,gpt-4o --judge_model gpt-4.1
```

**Open-weight models (or any OpenAI-compatible API)** via vLLM, SGLang, Ollama, etc.:

```bash
vllm serve Qwen/Qwen3-32B --port 8000

python run.py --task f1-dnf --provider openai --base_url http://localhost:8000/v1 --api_key EMPTY \
    --models Qwen/Qwen3-32B,Qwen/Qwen3-32B,Qwen/Qwen3-32B --judge_model Qwen/Qwen3-32B
```

Notes:
- With `--provider openai`, `--models` (3 planner IDs) and `--judge_model` are required,
  since the per-task defaults are Claude model IDs.
- Planner *k* uses temperature `temp + 0.2k` to diversify plans. Models that do not accept a
  temperature (e.g. OpenAI reasoning models) fall back to their default, with a warning.
- The judge writes runnable Python, so a strong code model as `--judge_model` matters most.
- Results reported in the paper were obtained with Claude; other backbones may perform differently.
- To add another provider, add a branch to `make_client` and `ask_model` in `refuge/llm.py`
  that returns the response text; the pipeline needs nothing else.

### 📁 Code structure

```
run.py              CLI entry point
refuge/
  tasks.py          per-task settings (entity table/key, target, evaluator columns, models)
  pipeline.py       main loop: table → column selection → 9 plans → judge → code → validation filter
  prompts.py        prompt templates for every step
  evaluate.py       LightGBM validation-AUROC check
  parsing.py        parsing of plans, judge JSON and generated code
  llm.py            LLM API wrapper (Anthropic / OpenAI-compatible)
data/rel-*/         schema, column statistics and task description given to the LLM
```

To add a task, add a `TaskConfig` entry in `refuge/tasks.py` (schema/statistics/task
files for rel-hm, rel-stack and rel-trial are already under `data/`).

### 📝 Citation

If you find ReFuGe useful in your research, please cite our paper:


```bibtex
@inproceedings{kim2026refuge,
  title     = {ReFuGe: Feature Generation for Prediction Tasks on Relational Databases with LLM Agents},
  author    = {Kim, Kyungho and Lee, Geon and Kim, Juyeon and Choi, Dongwon and Kang, Shinhwan and Shin, Kijung},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year      = {2026},
  publisher = {ACM},
  eprint    = {2601.17735},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2601.17735}
}
```

