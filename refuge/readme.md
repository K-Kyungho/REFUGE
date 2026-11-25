# refuge

The ReFuGe pipeline, shared by all tasks. `run.py` at the repository root is the
command-line entry point; it picks a `TaskConfig`, builds an LLM client and calls
`pipeline.run`.

| file | contents |
|---|---|
| `pipeline.py` | main loop: table selection → column selection → 9 feature plans → judge picks 3 → code generation and execution → validation filter |
| `tasks.py` | `TaskConfig` and the `TASKS` registry (entity table/key, target, evaluator columns, data files, planner models) |
| `prompts.py` | prompt templates for every step, and the code skeleton given to the judge |
| `llm.py` | `make_client` and `ask_model`, the only code that talks to the Anthropic / OpenAI SDKs |
| `parsing.py` | parsing of feature plans, the judge's JSON and generated code |
| `evaluate.py` | LightGBM validation AUROC used to accept or reject each feature |

## Extending

- **New task:** add a `TaskConfig` entry to `TASKS` in `tasks.py`, with its schema,
  statistics and task files under `data/` (see `data/readme.txt`).
- **New LLM provider:** add a branch to `make_client` and `ask_model` in `llm.py`
  that returns the response text.
