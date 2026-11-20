#!/usr/bin/env python
"""ReFuGe entry point.

    python run.py --task f1-dnf
    python run.py --task avito-clicks --n_iter 3 --out_dir outputs/avito-clicks
"""

import argparse
import os
import random

os.environ.setdefault("OMP_NUM_THREADS", "8")

import anthropic  # noqa: E402

from refuge.pipeline import run  # noqa: E402
from refuge.tasks import TASKS  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--n_iter", type=int, default=3)
    p.add_argument("--temp", type=float, default=0.2)
    p.add_argument("--delta", type=float, default=0.0,
                   help="keep a feature only if val AUROC >= best + delta")
    p.add_argument("--models", type=str, default=None,
                   help="comma-separated 3 planner model IDs (default: per-task setting)")
    p.add_argument("--judge_model", type=str, default="claude-sonnet-4-20250514",
                   help="selects 3 of the 9 plans and writes the feature code")
    p.add_argument("--data_dir", type=str, default=os.path.join(ROOT, "data"))
    p.add_argument("--out_dir", type=str, default=None,
                   help="default: outputs/<dataset>/<task>")
    p.add_argument("--seed", type=int, default=None,
                   help="seed for the evaluator's train subsampling (default: unseeded)")
    p.add_argument("--api_key", type=str, default=None,
                   help="default: $ANTHROPIC_API_KEY")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    cfg = TASKS[args.task]
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    out_dir = args.out_dir or os.path.join(ROOT, "outputs", cfg.dataset, cfg.task)
    client = anthropic.Anthropic(api_key=args.api_key) if args.api_key else anthropic.Anthropic()

    run(cfg, client, data_dir=args.data_dir, out_dir=out_dir, n_iter=args.n_iter,
        temp=args.temp, delta=args.delta, planner_models=models, judge_model=args.judge_model)


if __name__ == "__main__":
    main()
