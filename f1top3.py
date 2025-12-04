#!/usr/bin/env python
# -*- coding: utf-8 -*-
# kkyungho@kaist.ac.kr


import os
os.environ['XDG_CACHE_HOME'] = '/data/cache'
os.environ['OMP_NUM_THREADS'] = "8"

import json
import argparse
import datetime
import pandas as pd
import lightgbm as lgb
import subprocess
import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import early_stopping, log_evaluation
import time
import copy
import re
import random
import string

# NEW: anthropic SDK
import anthropic

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.base import TaskType
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder  # 안정적
from scipy.stats import pointbiserialr
from sklearn.feature_selection import mutual_info_classif
import optuna
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
#import lightgbm as lgb
#from lightgbm import early_stopping, log_evaluation

# ======================
# Small helpers
# ======================

def load_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def to_single_text_block(resp):
    texts = []
    for b in resp:
        if hasattr(b, "text"):
            texts.append(b.text)
        else:
            texts.append(str(b))
    return [{"type": "text", "text": "\n".join(texts)}]

def block_to_str(block_or_list) -> str:
    """Accepts list[{'type':'text','text':...}] or str and returns str."""
    if isinstance(block_or_list, list):
        out = []
        for b in block_or_list:
            if isinstance(b, dict) and "text" in b:
                out.append(b["text"])
            else:
                out.append(str(b))
        return "\n".join(out)
    return str(block_or_list)

def ask_model(client, model_id: str, base_msgs: list, user_text: str, system_text: str, max_tokens=5000, temperature=0.2):
    resp = client.messages.create(
        model=model_id,
        system=system_text,
        messages=base_msgs + [{"role": "user", "content": [{"type": "text", "text": user_text}]}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return to_single_text_block(resp.content), resp.content  # (normalized_block, raw_content)

_FEATURE_PAT = re.compile(
    r"Feature\s*\d+\s*:\s*"
    r"(?:-+\s*)?"
    r"(?:Name\s*:\s*(?P<name>.+?)\n)?"
    r"(?:.*?Plan\s*:\s*(?P<plan>.+?))"
    r"(?:\n-?\s*Rationale\s*:\s*(?P<rat>.+?))?(?:\n\n|\Z)",
    flags=re.IGNORECASE | re.DOTALL
)

def parse_feature_plans(step3_block, iter_idx: int, src_model: str) -> List[Dict[str, Any]]:
    """
    Step 3 응답에서 최대 3개 (Name/Plan/Rationale) 블록을 추출.
    완벽히 못 잡더라도 Plan이 있으면 후보로 수집.
    """
    txt = block_to_str(step3_block)
    items = []
    for m in _FEATURE_PAT.finditer(txt):
        name = (m.group("name") or f"feat{iter_idx}_?").strip()
        plan = (m.group("plan") or "").strip()
        rat  = (m.group("rat")  or "").strip()
        if plan:
            items.append({
                "iter": iter_idx,
                "src_model": src_model,
                "name_raw": name,
                "plan": plan,
                "rationale": rat,
            })
    # 세이프가드: 아무것도 못 잡았을 때 'Plan:' 라인 근처라도 수집
    if not items:
        chunks = re.split(r"\n\s*\n", txt)
        for ch in chunks:
            if "plan" in ch.lower():
                items.append({
                    "iter": iter_idx,
                    "src_model": src_model,
                    "name_raw": f"feat{iter_idx}_?",
                    "plan": ch.strip(),
                    "rationale": "",
                })
    return items[:3]

def fewshot_block_simple(preds_df, entity_col: str, n_each: int = 3, max_cols: int = 10):
    df = preds_df.copy()
    df["__margin__"] = np.abs(df["y_prob"] - 0.5)
    seed = random.randint(0, 10**6)
    correct   = df[df["y_true"] == df["y_pred"]].sample(n=min(n_each, len(df[df["y_true"] == df["y_pred"]])), random_state=seed)
    incorrect = df[df["y_true"] != df["y_pred"]].sample(n=min(n_each, len(df[df["y_true"] != df["y_pred"]])), random_state=seed)

    def pick_cols(d):
        base = [c for c in d.columns if c not in {"y_true","y_pred","y_prob","__margin__"}]
        if entity_col in base:
            base.remove(entity_col)
            base = [entity_col] + base
        return base[:max_cols]

    def fmt_block(d, title):
        if len(d) == 0:
            return f"{title}\n  (none)"
        cols = pick_cols(d)
        show = cols + [c for c in ["y_true","y_pred","y_prob"] if c not in cols]
        lines = [title]
        for _, row in d[show].iterrows():
            parts = [f"{c}={row[c] if not isinstance(row[c], float) else float(row[c]):.4g}" 
                     if isinstance(row[c], float) else f"{c}={row[c]}" for c in show]
            lines.append("  • " + ", ".join(parts))
        return "\n".join(lines)

    out = [
        "Prediction-based validation examples:",
        fmt_block(correct,   "CORRECT (y_pred == y_true):"),
        fmt_block(incorrect, "INCORRECT (y_pred != y_true):")
    ]
    return "\n".join(out)

def feature_corr_mi(
    train_df: pd.DataFrame,
    target_col: str,
    key_col: str
) -> str:
    candidate_cols = [c for c in train_df.columns]
    feature_col = candidate_cols[-1]
    return str(feature_col)


# ======================
# Main
# ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="YOUR API KEY")
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--task", type=str, default="driver-top3")
    parser.add_argument("--schema_path_gt", type=str, default="rel-f1/f1_schema.txt")
    parser.add_argument("--stats_path", type=str, default="rel-f1/feature_f1.txt")
    parser.add_argument("--task_path", type=str, default="rel-f1/task-driver-top3.txt")
    parser.add_argument("--entity_table", type=str, default="drivers")
    parser.add_argument("--entity_col", type=str, default="driverId")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--n_iter", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.0)

    parser.add_argument(
        "--models",
        type=str,
        default="claude-sonnet-4-20250514, claude-sonnet-4-20250514, claude-sonnet-4-20250514",
        help="Comma-separated model IDs for the 3 planner LLMs (will be used to generate 3 plans each)"
    )
    parser.add_argument(
        "--voter_model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="LLM judge to select final 3 features among 9 candidates"
    )

    args = parser.parse_args()
    client = anthropic.Client(api_key=args.api_key)

    # cumulative feature frames
    cum_train = None
    cum_valid = None
    cum_test = None

    ui_train_ = None
    ui_valid_ = None
    ui_test_ = None

    # load texts
    db_schema_gt     = load_text_file(args.schema_path_gt)
    db_stats         = load_text_file(args.stats_path)
    task_description = load_text_file(args.task_path)

    dataset     = args.dataset
    task        = args.task
    entity_col  = args.entity_col
    entity_table= args.entity_table

    dataset_relbench = get_dataset(args.dataset)
    task_relbench = get_task(args.dataset, args.task, TaskType.BINARY_CLASSIFICATION)
    
    def check_feature_k(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        target_col: str = "qualifying",
        drop_cols: tuple = ('date', 'driverRef', 'code', 'forename', 'surname', 'dob'),
        sample_frac: float = 0.3,
        entity_col_local: str = "Id"
    ) -> dict:
        seed = random.randint(0, 10**6)
        train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")].copy()
        valid_df = valid_df.loc[:, ~valid_df.columns.str.contains("^Unnamed")].copy()
        train_df.fillna(0, inplace=True)
        valid_df.fillna(0, inplace=True)

        train_df = train_df.sample(frac=sample_frac, random_state=seed)

        X_train = train_df.drop(columns=list(drop_cols) + [target_col], errors="ignore")
        y_train = train_df[target_col].astype(int)
        X_val = valid_df.drop(columns=list(drop_cols) + [target_col], errors="ignore")
        y_val = valid_df[target_col].astype(int)

        categorical_cols = ['nationality']

        if categorical_cols:
            for col in categorical_cols:
                le = LabelEncoder()
                le.fit(X_train[col].astype(str))
                mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
                X_train[col] = X_train[col].astype(str).map(mapping).fillna(-1).astype(int)
                X_val[col] = X_val[col].astype(str).map(mapping).fillna(-1).astype(int)

        lgb_params = dict(
            objective="binary",
            metric="auc",
            boosting_type="gbdt",
            learning_rate=0.05,
            feature_fraction=0.8,
            max_depth=-1,
            num_leaves=10,
            verbose=-1,
        )
        lgb_tr = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(
            params=lgb_params,
            train_set=lgb_tr,
            valid_sets=[lgb_val],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(40)],
        )
        y_prob = model.predict(X_val, num_iteration=model.best_iteration)
        auroc = roc_auc_score(y_val, y_prob)

        preds_df = pd.DataFrame({
            "y_true": y_val.values.astype(int),
            "y_prob": y_prob,
        })
        preds_df["y_pred"] = (preds_df["y_prob"] >= 0.5).astype(int)

        kept_cols = [c for c in valid_df.columns if (c not in drop_cols and c != target_col)]
        if entity_col_local is not None and entity_col_local not in kept_cols and entity_col_local in valid_df.columns:
            kept_cols = [entity_col_local] + kept_cols
        preds_df = pd.concat(
            [valid_df[kept_cols].reset_index(drop=True), preds_df.reset_index(drop=True)],
            axis=1
        )
        return float(auroc), preds_df

    def merge_feat(base_df, new_df):
        key = args.entity_col
        df_out = base_df.merge(new_df, on=key, how='left')
        print(df_out.shape)
        return df_out

    db = dataset_relbench.get_db()
    ui = dataset_relbench.get_db().table_dict[entity_table].df

    ui_train = task_relbench.get_table("train").df
    ui_valid = task_relbench.get_table("val", mask_input_cols=False).df
    ui_test  = task_relbench.get_table("test", mask_input_cols=False).df

    ra_dir = f"data/{args.dataset}/{args.task}/"
    os.makedirs(ra_dir, exist_ok=True)
    num = 0

    def merge_files(feature_files, base_csv, key=args.entity_col):
        df_out = pd.read_csv(base_csv)
        for fpath in feature_files:
            df = pd.read_csv(fpath)
            base_cols = set(df_out.columns)
            new_cols = [c for c in df.columns if c not in base_cols]
            df_feat = df[[key] + new_cols]
            df_out = df_out.merge(df_feat, on=key, how='left')
        return df_out

    # === conversational base context
    base_msgs=[
        {"role": "assistant", "content": f"RDB Schema: RDB Schema with Pkey-Fkey {db_schema_gt}"},
        {"role": "assistant", "content": f"RDB Statistics: We also have stats for all columns in RDB {db_stats}"},
        {"role": "assistant", "content": f"Task: And the task we want to solve is :{task_description}"},
        {"role": "assistant", "content": f"Our goal is to create features that are most helpful for solving the given task."}
    ]

    kept = []
    iter_cols = {}
    best_auc = 0.0

    train_csv_ori_path = f"data/{dataset}/{task}-train-ori.csv"
    valid_csv_ori_path = f"data/{dataset}/{task}-val-ori.csv"
    test_csv_ori_path  = f"data/{dataset}/{task}-test-ori.csv"

    base_train = pd.read_csv(train_csv_ori_path, index_col = False)
    base_valid = pd.read_csv(valid_csv_ori_path, index_col = False)
    base_test  = pd.read_csv(test_csv_ori_path,  index_col = False)

    try:
        auroc_prev, preds_df = check_feature_k(base_train, base_valid)
    except Exception as e:
        print(f"[WARN] baseline check_feature failed: {e}")
        auroc_prev = 0.5

    def merge_on_entity(base_df, add_df, key):
        return base_df.merge(add_df, on=key, how='left')

    # ------------- ITER LOOP -------------
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(model_ids) != 3:
        print(f"[WARN] --models expects 3 models; got {len(model_ids)}. It will still run with provided list.")

    for i in range(1, args.n_iter + 1):
        time.sleep(1)
        # -------- Step 1 --------
        step1_prompt = f"""
You are in **STEP 1.1: Table Selection & Join Paths**.

Goal: Given the task description and target, select helper tables that would help solve the task and provide explicit join paths from TABLE: {entity_table} COL: {entity_col} to each helper table using PK/FK in the schema.

### TASK
{task_description}

### TARGET
- target_table: {entity_table}
- target_key:   {entity_col}

### Output Format
- TABLE 1
- TABLE 2
- TABLE 3
"""
        if i > 1:
            step1_prompt += (
                f"\nNOTE: This is iteration {i}. "
                f"Carefully review the helper tables and join paths that were already selected in previous iteration(s). "
                f"Do NOT repeat or trivially modify them. "
                f"Instead, suggest alternative or complementary tables and join paths "
                f"that could generate new, non-redundant features and provide additional predictive signal."
            )

        resp1_block, resp1_raw = ask_model(
            client, model_ids[0], base_msgs, step1_prompt,
            "You are a data scientist for prediction task in relational database.", max_tokens=5000, temperature=args.temp
        )
        response_step1 = resp1_block
        print("=== Find the Table ===")
        print(response_step1, "\n")
        base_msgs.append({"role": "assistant", "content": response_step1})

        # -------- Step 1.2 --------
        step2_prompt = f"""
Now you are in **STEP 1.2: Column Selection**.

In the previous step, you choose the tables that would be helpful in the task.
---Response in Step 1---
{resp1_raw}
------

From these tables, please select columns that would be helpful for the task. We should generate new columns from aggregating these columns.

# IMPORTANT
- Columns will be used for calculating new feature to augment {entity_table}.
- Reference columns that exist in the schema.
- Output a concise bulleted list of columns and tables, each with a short rationale.
"""
        resp2_block, resp2_raw = ask_model(
            client, model_ids[0], base_msgs, step2_prompt,
            "You are a data scientist for prediction task in relational database.", max_tokens=5000, temperature=args.temp
        )
        response_step2 = resp2_block
        print("=== Find the Columns ===")
        print(response_step2, "\n")
        base_msgs.append({"role": "assistant", "content": response_step2})

        step3_prompt_template = f"""
You are now in **STEP 2: Feature Generation**.

Goal: Based on the responses from Step 1.1 (Tables) and Step 1.2 (Columns), decide which aggregation functions to apply in order to generate up to three new features that augment the target table {entity_table}.

--- Response from Step 1 ---
{block_to_str(response_step1)}
------

--- Response from Step 2 ---
{block_to_str(response_step2)}
------

# IMPORTANT
- Select up to three features to create.
- Each feature must be derived by applying a clear aggregation function (e.g., COUNT, SUM, AVG, MAX, MIN, STD).
- Each feature must be numeric or categorical (float or int).
- Each feature must join back to the target table on {entity_col}.
- Exactly one value per {entity_col}.
- Provide a short rationale for why this aggregation is useful for the task.
- Output a concise bulleted list of the final features.
- Avoid column-name collisions on joins.

### Output Format (strict)
Feature 1:
- Name: feat{i}_1
- Plan: ...
- Rationale: ...

Feature 2:
- Name: feat{i}_2
- Plan: ...
- Rationale: ...

Feature 3:
- Name: feat{i}_3
- Plan: ...
- Rationale: ...
"""

        all_plans = []
        temp = args.temp
        for mi, mid in enumerate(model_ids):
            #temp += 0.2
            step3_block_m, step3_raw_m = ask_model(
                client, mid, base_msgs, step3_prompt_template,
                "You are a data scientist for prediction task in relational database.",
                max_tokens=5000, temperature=temp
            )
            temp += 0.2
            print(f"=== Make Features (model={mid}) ===")
            print(step3_block_m, "\n")
            base_msgs.append({"role":"assistant","content": step3_block_m})

            parsed = parse_feature_plans(step3_block_m, iter_idx=i, src_model=mid)
            # 최대 3개 수집
            all_plans.extend(parsed)

        if len(all_plans) == 0:
            raise RuntimeError("No feature plans parsed from any model in Step 3.")

        def compact_plan(p: Dict[str, Any], idx: int) -> str:
            return json.dumps({
                "id": idx,
                "model": p["src_model"],
                "iter": p["iter"],
                "name": p.get("name_raw", f"feat{i}_?"),
                "plan": p["plan"],
                "rationale": p.get("rationale","")
            }, ensure_ascii=False)

        numbered = [compact_plan(p, j+1) for j, p in enumerate(all_plans)]
        ### Step 3.1 Reasoning-based Feature Filtering
        voter_prompt = f"""
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
{chr(10).join(numbered)}
"""
        voter_block, voter_raw = ask_model(
            client, args.voter_model, base_msgs, voter_prompt,
            "You are a rigorous judge for feature plan selection in relational databases.",
            max_tokens=2000, temperature=0.0
        )
        voter_text = block_to_str(voter_block)
        print("=== [VOTER RESULT] Raw ===")
        print(voter_text, "\n")

        selected_ids = []
        reasons = {}
        try:
            m = re.search(r"\{[\s\S]*\}", voter_text)
            if m:
                obj = json.loads(m.group(0))
                selected_ids = obj.get("selected_ids", [])
                reasons = obj.get("reasons", {})
        except Exception as e:
            print("[WARN] voter JSON parse failed; falling back to first 3.")
            selected_ids = [1,2,3]
            reasons = {}

        if not selected_ids or len(selected_ids) < 3:
            print("[WARN] voter returned insufficient ids; falling back to first 3.")
            selected_ids = [1,2,3]

        selected_plans = []
        for sid in selected_ids[:3]:
            idx0 = int(sid) - 1
            if 0 <= idx0 < len(all_plans):
                selected_plans.append(all_plans[idx0])
        std_names = [f"feat{i}_1", f"feat{i}_2", f"feat{i}_3"]
        for p, std in zip(selected_plans, std_names):
            p["name_std"] = std

        print("=== [VOTER RESULT] Selected 3 Feature Plans ===")
        for p in selected_plans:
            rid = all_plans.index(p) + 1
            why = reasons.get(str(rid), reasons.get(rid, ""))
            print(f"[id={rid}] {p['name_std']} from {p['src_model']}")
            print("PLAN:", p["plan"][:400], "...\nRATIONALE:", (p.get('rationale','') or "")[:200])
            if why:
                print("JUDGE_REASON:", why[:300])
            print()

        def _fmt_plan_block(p):
            return f"""- Name: {p['name_std']}
- Plan: {p['plan']}
- Rationale: {p.get('rationale','')}
"""
        feature_plan_block_for_step4 = "\n".join(_fmt_plan_block(p) for p in selected_plans)

        code_skeleton = f"""
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
        parser.add_argument("--dataset", type=str, default='{dataset}')
        parser.add_argument("--task", type=str, default='{task}')
        args = parser.parse_args()

        dataset = get_dataset(args.dataset)
        task = get_task(args.dataset, args.task, TaskType.BINARY_CLASSIFICATION)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        db = dataset.get_db()
        ui = dataset.get_db().table_dict["{entity_table}"].df

        ui_train = task.get_table("train").df
        ui_val = task.get_table("val", mask_input_cols=False).df
        ui_test = task.get_table("test", mask_input_cols=False).df

        timemax_trn = ui_train["date"].max() #**DO NOT MODIFY THIS TIMESTAMP**
        timemax_val = ui_val["date"].max()
        timemax_tst = ui_test["date"].max()

        print(timemax_trn)
        print(timemax_val)
        print(timemax_tst)

        # ====== BEGIN FEATURE FUNCTIONS ======
        # def compute_feature1(db, setting): ...
        # def compute_feature2(db, setting): ...
        # def compute_feature3(db, setting): ...
        # ====== END FEATURE FUNCTIONS ======

        os.makedirs("../../../data/{dataset}/{task}/", exist_ok=True)


        try:
            df_trn1 = compute_feature1(db, setting='train')
            df_val1 = compute_feature1(db, setting='validation')
            df_tst1 = compute_feature1(db, setting='test')
            df_trn1.to_csv("data/Claude/{dataset}/{task}/{task}-train-{i}-1.csv", index=False)
            df_val1.to_csv("data/Claude/{dataset}/{task}/{task}-val-{i}-1.csv", index=False)
            df_tst1.to_csv("data/Claude/{dataset}/{task}/{task}-test-{i}-1.csv", index=False)
        except Exception as e:
            print("[feat1-error]", e)

        try:
            df_trn2 = compute_feature2(db, setting='train')
            df_val2 = compute_feature2(db, setting='validation')
            df_tst2 = compute_feature2(db, setting='test')
            df_trn2.to_csv("data/Claude/{dataset}/{task}/{task}-train-{i}-2.csv", index=False)
            df_val2.to_csv("data/Claude/{dataset}/{task}/{task}-val-{i}-2.csv", index=False)
            df_tst2.to_csv("data/Claude/{dataset}/{task}/{task}-test-{i}-2.csv", index=False)
        except Exception as e:
            print("[feat2-error]", e)

        try:
            df_trn3 = compute_feature3(db, setting='train')
            df_val3 = compute_feature3(db, setting='validation')
            df_tst3 = compute_feature3(db, setting='test')
            df_trn3.to_csv("data/Claude/{dataset}/{task}/{task}-train-{i}-3.csv", index=False)
            df_val3.to_csv("data/Claude/{dataset}/{task}/{task}-val-{i}-3.csv", index=False)
            df_tst3.to_csv("data/Claude/{dataset}/{task}/{task}-test-{i}-3.csv", index=False)
        except Exception as e:
            print("[feat3-error]", e)

        print("Dataframes saved successfully!")
        """

        step4_prompt = f"""
Based on the three feature plans below, write a Python script **defining three separate functions** (compute_feature1, compute_feature2, compute_feature3).
Each should return a DataFrame with columns ['{entity_col}', {std_names[0]}], ['{entity_col}', {std_names[1]}], ['{entity_col}', {std_names[2]}].
At the end, save each feature for train/val/test as a separate CSV file. Follow this format strictly (all three functions must be implemented, no narrative):

### OUTPUT INSTRUCTIONS  (read **carefully**)
1. **Copy the entire skeleton exactly as shown, do NOT remove any line.**
2. Replace each ‘PLEASE FILL IN BELOW FUNCTION’ section with working code.
3. The first line of your reply must be “```python” and the last line must be “```”.
4. No narrative, no explanation — code only.
5. Each function should return a DataFrame with columns ONLY **['{entity_col}', {std_names[0]}]**, **['{entity_col}', {std_names[1]}]**, **['{entity_col}', {std_names[2]}]**.
6. Do not use columns or tables that do not exist in the schema.
7. If there are any "timestamp" column type in a table, prevent data leakage by filtering rows using the appropriate cutoff (timemax_trn/val/tst) for the given split.
8. Ensure you produce exactly one value per {entity_col} (i.e. one row per entity).
9. **When joining/merging two tables that may share column names, you MUST proactively disambiguate to avoid unintended overwrites.**
   - In pandas: always pass an explicit `suffixes=('<left>','<right>')` to `pd.merge(...)`, or pre-rename columns before merging; never rely on ambiguous `_x/_y` silently.
   - Select only the necessary columns for the join/output (project columns explicitly) so the result has **unique, meaningful column names**.
10. After any join, drop/rename overlapping columns that are not needed so that the final returned DataFrame contains **only** `['{entity_col}', <feat...>]` and **no duplicate or ambiguous names**.

### Final Selected Feature Plans (3)
{feature_plan_block_for_step4}

---Code Skeleton---
{code_skeleton}
---Done---
"""

        # 코드 생성은 안전하게 voter 또는 첫 모델 중 하나 택1
        winner_model_for_code = args.voter_model

        resp4 = client.messages.create(
            model=winner_model_for_code,
            system="You are a data scientist for prediction task in relational database.",
            messages=base_msgs + [{"role": "user", "content": [{"type": "text", "text": step4_prompt}]}],
            max_tokens=5000,
            temperature=args.temp
        )

        if isinstance(resp4.content, list):
            full_response = ""
            for block in resp4.content:
                if hasattr(block, "text"):
                    full_response += block.text
                else:
                    full_response += str(block)
        else:
            full_response = resp4.content

        m = re.search(r"```python\s*(.*?)\s*```", full_response, re.S)
        code_only = m.group(1) if m else full_response

        output_file = f'../Claude3.7_{args.dataset}_{args.task}{args.n_iter}-1115-iter{i}.py'
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(code_only)
        print(f"[INFO] Final code saved to: {output_file}")

        try:
            subprocess.run(["python", f'{output_file}'], check=True)
        except Exception as e:
            print(f"[WARN] Generated feature code execution failed (some or all). Will salvage any produced files: {e}")

        kept_in_iter = False
        for k in range(1, 4):
            train_path = f"data/{task}-train-{i}-{k}.csv"
            val_path   = f"data/{task}-val-{i}-{k}.csv"
            test_path  = f"data/{task}-test-{i}-{k}.csv"

            if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
                train_csv_path = train_path
                valid_csv_path = val_path
                test_csv_path  = test_path
            else:
                print(f"[INFO] Missing outputs for k={k}: {train_path} | {val_path} | {test_path}")
                continue

            df_trn = pd.read_csv(train_csv_path, index_col=False)
            df_val = pd.read_csv(valid_csv_path, index_col=False)
            df_tst = pd.read_csv(test_csv_path, index_col=False)

            df_join_trn = pd.merge(ui, df_trn, on=entity_col, how='left')
            df_join_val = pd.merge(ui, df_val, on=entity_col, how='left')
            df_join_tst = pd.merge(ui, df_tst, on=entity_col, how='left')

            if ui_train_ is None:
                train_df = pd.merge(ui_train, df_join_trn, on=entity_col, how='left')
                valid_df = pd.merge(ui_valid, df_join_val, on=entity_col, how='left')
                test_df  = pd.merge(ui_test,  df_join_tst, on=entity_col, how='left')
            else:
                train_df_ = pd.merge(ui_train_, df_trn, on=entity_col, how='left')
                valid_df_ = pd.merge(ui_valid_, df_val, on=entity_col, how='left')
                test_df_  = pd.merge(ui_test_,  df_tst, on=entity_col, how='left')

                train_df = pd.merge(ui_train, train_df_, on=entity_col, how='left')
                valid_df = pd.merge(ui_valid, valid_df_, on=entity_col, how='left')

            new_auc, preds_df = check_feature_k(train_df, valid_df)

            print(f"new AUROC: {new_auc:.4f}")

            if new_auc >= auroc_prev:
                kept_in_iter = True
                num += 1
                print(f"Remain Feature in round {i}-{k}, improved AUROC {auroc_prev:.4f} -> {new_auc:.4f}")

                resp_succ_text = (
                    f"FEEDBACK: Feature '{feature_col}' significantly improved AUC from {auroc_prev:.4f} to {new_auc:.4f}."
                )

                print("=== Success Reason ===")
                print(resp_succ_text, "\n")
                auroc_prev = new_auc
                base_msgs.append({"role":"assistant","content": resp_succ_text})

                if cum_train is None:
                    cum_train = merge_feat(ui, df_trn)
                    cum_valid = merge_feat(ui, df_val)
                    cum_test  = merge_feat(ui, df_tst)

                    ui_train_ = merge_feat(ui, df_trn)
                    ui_valid_ = merge_feat(ui, df_val)
                    ui_test_  = merge_feat(ui, df_tst)
                else:
                    ui_train_ = merge_feat(cum_train, df_trn)
                    ui_valid_ = merge_feat(cum_valid, df_val)
                    ui_test_  = merge_feat(cum_test,  df_tst)

                    cum_train = merge_feat(cum_train, df_trn)
                    cum_valid = merge_feat(cum_valid, df_val)
                    cum_test  = merge_feat(cum_test,  df_tst)

            else:
                resp_fail_text = (
                    f"FEEDBACK: Feature '{feature_col}' caused the AUC to drop from {auroc_prev:.4f} to {new_auc:.4f}."
                )
                print("=== Fail Reason ===")
                print(resp_fail_text, "\n")
                base_msgs.append({"role":"assistant","content": resp_fail_text})
        if not kept_in_iter:
            print(f"[INFO] No features were kept in iteration {i}; stopping early (max_iter={args.n_iter}).")
            break
    print("Merging Features")

    df_train = pd.merge(ui_train, cum_train, on=entity_col, how='left')
    df_val   = pd.merge(ui_valid, cum_valid, on=entity_col, how='left')
    df_tst   = pd.merge(ui_test,  cum_test,  on=entity_col, how='left')

    df_train.to_csv(os.path.join(ra_dir, f"{args.task}-train-{args.n_iter}.csv"), index=False)
    df_val.to_csv(os.path.join(ra_dir, f"{args.task}-valid-{args.n_iter}.csv"), index=False)
    df_tst.to_csv(os.path.join(ra_dir, f"{args.task}-test-{args.n_iter}.csv"), index=False)
    
    print(f"Final CSVs saved with {num} kept features.")
    for idx in range(1, args.n_iter + 1):
        for split in ("train", "val", "test"):
            for k in (1, 2, 3):
                fname = f"{args.task}-{split}-{idx}-{k}.csv"
                fpath = os.path.join(ra_dir, fname)
                if os.path.exists(fpath):
                    print(fpath)

    print("Intermediate iteration files removed.")


if __name__ == "__main__":
    main()
