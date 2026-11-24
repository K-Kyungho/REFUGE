"""Prompt templates for each ReFuGe step (wording kept from the original scripts)."""

import json
from typing import Any, Dict, List

from .tasks import TaskConfig


def base_context(schema: str, stats: str, task_description: str) -> List[dict]:
    return [
        {"role": "assistant", "content": f"RDB Schema: RDB Schema with Pkey-Fkey {schema}"},
        {"role": "assistant", "content": f"RDB Statistics: We also have stats for all columns in RDB {stats}"},
        {"role": "assistant", "content": f"Task: And the task we want to solve is :{task_description}"},
        {"role": "assistant", "content": "Our goal is to create features that are most helpful for solving the given task."},
    ]


# ---------------------------------------------------------------- Step 1.1
def table_selection(cfg: TaskConfig, task_description: str, it: int) -> str:
    prompt = f"""
You are in **STEP 1.1: Table Selection & Join Paths**.

Goal: Given the task description and target, select helper tables that would help solve the task and provide explicit join paths from TABLE: {cfg.entity_table} COL: {cfg.entity_col} to each helper table using PK/FK in the schema.

### TASK
{task_description}

### TARGET
- target_table: {cfg.entity_table}
- target_key:   {cfg.entity_col}

### Output Format
- TABLE 1
- TABLE 2
- TABLE 3
"""
    if it > 1:
        prompt += (
            f"\nNOTE: This is iteration {it}. "
            f"Carefully review the helper tables and join paths that were already selected in previous iteration(s). "
            f"Do NOT repeat or trivially modify them. "
            f"Instead, suggest alternative or complementary tables and join paths "
            f"that could generate new, non-redundant features and provide additional predictive signal."
        )
    return prompt


# ---------------------------------------------------------------- Step 1.2
def column_selection(cfg: TaskConfig, step1_raw) -> str:
    # NOTE: the original scripts interpolate the *raw* SDK content list here
    # (i.e. the repr of the TextBlock objects); kept as-is for reproducibility.
    return f"""
Now you are in **STEP 1.2: Column Selection**.

In the previous step, you choose the tables that would be helpful in the task.
---Response in Step 1---
{step1_raw}
------

From these tables, please select columns that would be helpful for the task. We should generate new columns from aggregating these columns.

# IMPORTANT
- Columns will be used for calculating new feature to augment {cfg.entity_table}.
- Reference columns that exist in the schema.
- Output a concise bulleted list of columns and tables, each with a short rationale.
"""


# ---------------------------------------------------------------- Step 2
def feature_generation(cfg: TaskConfig, step1_text: str, step2_text: str, it: int) -> str:
    return f"""
You are now in **STEP 2: Feature Generation**.

Goal: Based on the responses from Step 1.1 (Tables) and Step 1.2 (Columns), decide which aggregation functions to apply in order to generate up to three new features that augment the target table {cfg.entity_table}.

--- Response from Step 1 ---
{step1_text}
------

--- Response from Step 2 ---
{step2_text}
------

# IMPORTANT
- Select up to three features to create.
- Each feature must be derived by applying a clear aggregation function (e.g., COUNT, SUM, AVG, MAX, MIN, STD).
- Each feature must be numeric or categorical (float or int).
- Each feature must join back to the target table on {cfg.entity_col}.
- Exactly one value per {cfg.entity_col}.
- Provide a short rationale for why this aggregation is useful for the task.
- Output a concise bulleted list of the final features.
- Avoid column-name collisions on joins.

### Output Format (strict)
Feature 1:
- Name: feat{it}_1
- Plan: ...
- Rationale: ...

Feature 2:
- Name: feat{it}_2
- Plan: ...
- Rationale: ...

Feature 3:
- Name: feat{it}_3
- Plan: ...
- Rationale: ...
"""


# ---------------------------------------------------------------- Step 3.1 (judge)
def judge(plans: List[Dict[str, Any]], it: int) -> str:
    numbered = [
        json.dumps({
            "id": j + 1,
            "model": p["src_model"],
            "iter": p["iter"],
            "name": p.get("name_raw", f"feat{it}_?"),
            "plan": p["plan"],
            "rationale": p.get("rationale", ""),
        }, ensure_ascii=False)
        for j, p in enumerate(plans)
    ]
    candidates = "\n".join(numbered)
    return f"""
You are the **JUDGE LLM**. We have N=9 candidate feature plans (JSON objects), produced by 3 different LLMs (3 each).
Pick the **best 3** that are (a) likely to be predictive for the task, (b) feasible to implement from the given schema/tables, and (c) not redundant with each other.
Favor central ideas (appear in multiple variants across models) while keeping diversity.

Return a **JSON** with:
{{
  "selected_ids": [id1, id2, id3],    // the 3 chosen candidate ids (1..9)
  "reasons": {{
      "id1": "<why chosen>",
      "id2": "<why chosen>",
      "id3": "<why chosen>"
  }}
}}

# Candidates (one per line):
{candidates}
"""


# ---------------------------------------------------------------- Step 3.2 (code)
def feature_csv_path(feat_dir: str, task: str, split: str, it: int, k: int) -> str:
    return f"{feat_dir}/{task}-{split}-{it}-{k}.csv"


def code_skeleton(cfg: TaskConfig, it: int, feat_dir: str) -> str:
    def load(var, split, masked):
        line = f'{var} = task.get_table("{split}"{", mask_input_cols=False" if masked else ""}).df'
        if cfg.task_col_rename:
            line += f"\n{var}.rename(columns={cfg.task_col_rename!r}, inplace=True)"
        return line

    save_blocks = []
    for k in (1, 2, 3):
        p = {s: feature_csv_path(feat_dir, cfg.task, s, it, k) for s in ("train", "val", "test")}
        save_blocks.append(f"""\
try:
    df_trn{k} = compute_feature{k}(db, setting='train')
    df_val{k} = compute_feature{k}(db, setting='validation')
    df_tst{k} = compute_feature{k}(db, setting='test')
    df_trn{k}.to_csv("{p['train']}", index=False)
    df_val{k}.to_csv("{p['val']}", index=False)
    df_tst{k}.to_csv("{p['test']}", index=False)
except Exception as e:
    print("[feat{k}-error]", e)
""")

    return f"""
import argparse
import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ['OMP_NUM_THREADS'] = "8"
import torch
import pandas as pd
import numpy as np

from relbench.base import TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='{cfg.dataset}')
parser.add_argument("--task", type=str, default='{cfg.task}')
args = parser.parse_args()

dataset = get_dataset(args.dataset)
task = get_task(args.dataset, args.task, TaskType.BINARY_CLASSIFICATION)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

db = dataset.get_db()
ui = dataset.get_db().table_dict["{cfg.entity_table}"].df

{load("ui_train", "train", False)}
{load("ui_val", "val", True)}
{load("ui_test", "test", True)}

timemax_trn = ui_train["{cfg.time_col}"].max() #**DO NOT MODIFY THIS LINE**
timemax_val = ui_val["{cfg.time_col}"].max()
timemax_tst = ui_test["{cfg.time_col}"].max()

print(timemax_trn)
print(timemax_val)
print(timemax_tst)

# ====== BEGIN FEATURE FUNCTIONS ======
# def compute_feature1(db, setting): ...
# def compute_feature2(db, setting): ...
# def compute_feature3(db, setting): ...
# ====== END FEATURE FUNCTIONS ======

os.makedirs("{feat_dir}", exist_ok=True)

{chr(10).join(save_blocks)}
print("Dataframes saved successfully!")
"""


def code_generation(cfg: TaskConfig, selected: List[Dict[str, Any]], std_names: List[str],
                    skeleton: str) -> str:
    plan_block = "\n".join(
        f"- Name: {p['name_std']}\n- Plan: {p['plan']}\n- Rationale: {p.get('rationale', '')}\n"
        for p in selected
    )
    e = cfg.entity_col
    return f"""
Based on the three feature plans below, write a Python script **defining three separate functions** (compute_feature1, compute_feature2, compute_feature3).
Each should return a DataFrame with columns ['{e}', {std_names[0]}], ['{e}', {std_names[1]}], ['{e}', {std_names[2]}].
At the end, save each feature for train/val/test as a separate CSV file. Follow this format strictly (all three functions must be implemented, no narrative):

### OUTPUT INSTRUCTIONS  (read **carefully**)
1. **Copy the entire skeleton exactly as shown, do NOT remove any line.**
2. Replace each ‘PLEASE FILL IN BELOW FUNCTION’ section with working code.
3. The first line of your reply must be “```python” and the last line must be “```”.
4. No narrative, no explanation — code only.
5. Each function should return a DataFrame with columns ONLY **['{e}', {std_names[0]}]**, **['{e}', {std_names[1]}]**, **['{e}', {std_names[2]}]**.
6. Do not use columns or tables that do not exist in the schema.
7. If there are any "timestamp" column type in a table, prevent data leakage by filtering rows using the appropriate cutoff (timemax_trn/val/tst) for the given split.
8. Ensure you produce exactly one value per {e} (i.e. one row per entity).
9. **When joining/merging two tables that may share column names, you MUST proactively disambiguate to avoid unintended overwrites.**
   - In pandas: always pass an explicit `suffixes=('<left>','<right>')` to `pd.merge(...)`, or pre-rename columns before merging; never rely on ambiguous `_x/_y` silently.
   - Select only the necessary columns for the join/output (project columns explicitly) so the result has **unique, meaningful column names**.
10. After any join, drop/rename overlapping columns that are not needed so that the final returned DataFrame contains **only** `['{e}', <feat...>]` and **no duplicate or ambiguous names**.

### Final Selected Feature Plans (3)
{plan_block}

---Code Skeleton---
{skeleton}
---Done---
"""
