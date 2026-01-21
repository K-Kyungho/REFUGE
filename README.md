# ReFuGe: Feature Generation for Prediction Tasks on Relational Databases with LLM Agents


This is the official implementation of ReFuGe **(Feature Generation for Prediction Tasks on Relational Databases with LLM Agents)**.

Accepted in **ACM WWW 2026**

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

Each dataset–task pair comes with its own Python script. To run experiments, simply execute the corresponding file (e.g., amazonchurn.py etc.).

```bash
python {dataset}{task}.py
```
