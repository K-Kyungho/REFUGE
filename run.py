#!/usr/bin/env python
"""ReFuGe entry point.

    python run.py --task f1-dnf
    python run.py --task avito-clicks --n_iter 3 --out_dir outputs/avito-clicks
"""

import argparse
import os
import random

os.environ.setdefault("OMP_NUM_THREADS", "8")

from refuge.llm import PROVIDERS, make_client  # noqa: E402
from refuge.pipeline import run  # noqa: E402
from refuge.tasks import TASKS  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-20250514"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--n_iter", type=int, default=3)
    p.add_argument("--temp", type=float, default=0.2)
    p.add_argument("--delta", type=float, default=0.0,
                   help="keep a feature only if val AUROC >= best + delta")
    p.add_argument("--models", type=str, default=None,
                   help="comma-separated 3 planner model IDs (default: per-task setting)")
    p.add_argument("--judge_model", type=str,
                   help=f"model that selects plans and writes feature code (Anthropic default: {DEFAULT_JUDGE_MODEL})")
    p.add_argument("--data_dir", type=str, default=os.path.join(ROOT, "data"))
    p.add_argument("--out_dir", type=str, default=None,
                   help="default: outputs/<dataset>/<task>")
    p.add_argument("--seed", type=int, default=None,
                   help="seed for the evaluator's train subsampling (default: unseeded)")
    p.add_argument("--provider", choices=PROVIDERS, default="anthropic",
                   help="LLM backend (default: anthropic)")
    p.add_argument("--base_url", help="custom API endpoint")
    p.add_argument("--api_key", type=str, default=None,
                   help="default: $ANTHROPIC_API_KEY or $OPENAI_API_KEY, depending on the provider")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    cfg = TASKS[args.task]
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    if args.provider == "openai":
        if models is None or len(models) != 3:
            p.error("--provider openai requires three comma-separated --models")
        if not args.judge_model or not args.judge_model.strip():
            p.error("--provider openai requires --judge_model")

    judge_model = DEFAULT_JUDGE_MODEL if args.judge_model is None else args.judge_model
    out_dir = args.out_dir or os.path.join(ROOT, "outputs", cfg.dataset, cfg.task)
    client = make_client(args.provider, api_key=args.api_key, base_url=args.base_url)

    run(cfg, client, data_dir=args.data_dir, out_dir=out_dir, n_iter=args.n_iter,
        temp=args.temp, delta=args.delta, planner_models=models, judge_model=judge_model)


if __name__ == "__main__":
    main()
