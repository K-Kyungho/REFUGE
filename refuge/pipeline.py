"""The ReFuGe feature-generation loop.

Per iteration:
  Step 1.1  table selection        (planner model 1)
  Step 1.2  column selection       (planner model 1)
  Step 2    feature plans          (3 planner models x 3 plans = 9 candidates)
  Step 3.1  reasoning-based filter (judge model picks 3)
  Step 3.2  code generation + run  (judge model writes compute_feature1..3)
  Step 3.3  validation filter      (keep a feature iff LightGBM val AUROC does not drop)
The loop stops early when an iteration keeps no feature.
"""

import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

import pandas as pd
from relbench.base import TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from . import prompts
from .evaluate import validation_auroc
from .llm import SYSTEM_DS, SYSTEM_JUDGE, ask_model, content_to_str
from .parsing import extract_python, parse_feature_plans, parse_judge
from .tasks import TaskConfig

SPLITS = ("train", "val", "test")


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_task_tables(cfg: TaskConfig):
    dataset = get_dataset(cfg.dataset)
    task = get_task(cfg.dataset, cfg.task, TaskType.BINARY_CLASSIFICATION)
    entity_df = dataset.get_db().table_dict[cfg.entity_table].df
    labels = {
        "train": task.get_table("train").df,
        "val": task.get_table("val", mask_input_cols=False).df,
        "test": task.get_table("test", mask_input_cols=False).df,
    }
    if cfg.task_col_rename:
        labels = {s: df.rename(columns=cfg.task_col_rename) for s, df in labels.items()}
    return entity_df, labels


def run(cfg: TaskConfig, client, *, data_dir: str, out_dir: str, n_iter: int = 3,
        temp: float = 0.2, delta: float = 0.0, planner_models: Optional[List[str]] = None,
        judge_model: str = "claude-sonnet-4-20250514") -> Dict[str, pd.DataFrame]:
    planner_models = list(planner_models or cfg.planner_models)
    if len(planner_models) != 3:
        print(f"[WARN] expected 3 planner models; got {len(planner_models)}.")

    E = cfg.entity_col
    out_dir = os.path.abspath(out_dir)
    feat_dir = os.path.join(out_dir, "features")
    code_dir = os.path.join(out_dir, "generated_code")
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(code_dir, exist_ok=True)

    task_description = _read(os.path.join(data_dir, cfg.task_file))
    history = prompts.base_context(
        schema=_read(os.path.join(data_dir, cfg.schema_file)),
        stats=_read(os.path.join(data_dir, cfg.stats_file)),
        task_description=task_description,
    )

    entity_df, labels = _load_task_tables(cfg)

    def evaluate(entity_level: Dict[str, pd.DataFrame]) -> float:
        return validation_auroc(
            labels["train"].merge(entity_level["train"], on=E, how="left"),
            labels["val"].merge(entity_level["val"], on=E, how="left"),
            target_col=cfg.target_col, drop_cols=cfg.drop_cols,
            categorical_cols=cfg.categorical_cols,
        )

    # Entity-level frames (entity table + accepted features), one per split.
    kept = {s: entity_df for s in SPLITS}
    try:
        best_auc = evaluate(kept)
    except Exception as e:
        print(f"[WARN] baseline evaluation failed: {e}")
        best_auc = 0.5
    print(f"Baseline AUROC: {best_auc:.4f}")

    n_kept = 0
    for it in range(1, n_iter + 1):
        time.sleep(1)

        # ---- Step 1.1: tables
        step1_block, step1_raw = ask_model(
            client, planner_models[0], history,
            prompts.table_selection(cfg, task_description, it), SYSTEM_DS, temperature=temp)
        print("=== Find the Table ===\n", step1_block, "\n")
        history.append({"role": "assistant", "content": step1_block})

        # ---- Step 1.2: columns
        step2_block, _ = ask_model(
            client, planner_models[0], history,
            prompts.column_selection(cfg, step1_raw), SYSTEM_DS, temperature=temp)
        print("=== Find the Columns ===\n", step2_block, "\n")
        history.append({"role": "assistant", "content": step2_block})

        # ---- Step 2: 3 planners x up to 3 plans
        gen_prompt = prompts.feature_generation(
            cfg, content_to_str(step1_block), content_to_str(step2_block), it)
        plans = []
        for mi, model_id in enumerate(planner_models):
            block, _ = ask_model(client, model_id, history, gen_prompt, SYSTEM_DS,
                                 temperature=temp + 0.2 * mi)
            print(f"=== Make Features (model={model_id}) ===\n", block, "\n")
            history.append({"role": "assistant", "content": block})
            plans.extend(parse_feature_plans(content_to_str(block), it, model_id))
        if not plans:
            raise RuntimeError("No feature plans parsed from any planner model.")

        # ---- Step 3.1: judge selects 3
        judge_block, _ = ask_model(client, judge_model, history, prompts.judge(plans, it),
                                   SYSTEM_JUDGE, max_tokens=2000, temperature=0.0)
        judge_text = content_to_str(judge_block)
        print("=== [JUDGE RESULT] Raw ===\n", judge_text, "\n")
        selected_ids, reasons = parse_judge(judge_text)

        selected = [plans[int(sid) - 1] for sid in selected_ids[:3]
                    if 0 <= int(sid) - 1 < len(plans)]
        std_names = [f"feat{it}_{k}" for k in (1, 2, 3)]
        for p, name in zip(selected, std_names):
            p["name_std"] = name

        print("=== [JUDGE RESULT] Selected Feature Plans ===")
        for p in selected:
            rid = plans.index(p) + 1
            print(f"[id={rid}] {p['name_std']} from {p['src_model']}")
            print("PLAN:", p["plan"][:400], "...\nRATIONALE:", (p.get("rationale") or "")[:200])
            why = reasons.get(str(rid), reasons.get(rid, ""))
            if why:
                print("JUDGE_REASON:", str(why)[:300])
            print()

        # ---- Step 3.2: code generation + execution
        skeleton = prompts.code_skeleton(cfg, it, feat_dir)
        code_block, _ = ask_model(
            client, judge_model, history,
            prompts.code_generation(cfg, selected, std_names, skeleton), SYSTEM_DS,
            temperature=temp)
        code_file = os.path.join(code_dir, f"iter{it}.py")
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(extract_python(content_to_str(code_block)))
        print(f"[INFO] Generated code saved to: {code_file}")
        try:
            subprocess.run([sys.executable, code_file], check=True)
        except Exception as e:
            print(f"[WARN] Generated code failed (partially or fully); using whatever was produced: {e}")

        # ---- Step 3.3: keep each feature iff validation AUROC does not drop
        kept_in_iter = False
        for k in (1, 2, 3):
            paths = {s: prompts.feature_csv_path(feat_dir, cfg.task, s, it, k) for s in SPLITS}
            if not all(os.path.exists(p) for p in paths.values()):
                print(f"[INFO] Missing outputs for feature {it}-{k}")
                continue
            feats = {s: pd.read_csv(p, index_col=False) for s, p in paths.items()}
            candidate = {s: kept[s].merge(feats[s], on=E, how="left") for s in SPLITS}
            feat_name = ", ".join(c for c in feats["train"].columns if c != E)

            try:
                new_auc = evaluate(candidate)
            except Exception as e:
                print(f"[WARN] evaluation of feature {it}-{k} failed; skipping: {e}")
                continue
            print(f"new AUROC: {new_auc:.4f}")

            if new_auc >= best_auc + delta:
                feedback = (f"FEEDBACK: Feature '{feat_name}' significantly improved AUC "
                            f"from {best_auc:.4f} to {new_auc:.4f}.")
                print(f"Kept feature {it}-{k}: AUROC {best_auc:.4f} -> {new_auc:.4f}")
                kept, best_auc = candidate, new_auc
                kept_in_iter = True
                n_kept += 1
            else:
                feedback = (f"FEEDBACK: Feature '{feat_name}' caused the AUC to drop "
                            f"from {best_auc:.4f} to {new_auc:.4f}.")
            print(feedback, "\n")
            history.append({"role": "assistant", "content": feedback})

        if not kept_in_iter:
            print(f"[INFO] No features kept in iteration {it}; stopping early.")
            break

    # ---- Save task tables augmented with all accepted features
    final = {s: labels[s].merge(kept[s], on=E, how="left") for s in SPLITS}
    for s, df in final.items():
        df.to_csv(os.path.join(out_dir, f"{cfg.task}-{s}.csv"), index=False)
    print(f"Final CSVs saved to {out_dir} with {n_kept} kept features (best val AUROC {best_auc:.4f}).")
    return final
