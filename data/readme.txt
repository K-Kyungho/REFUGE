Text inputs given to the LLM for each RelBench dataset.

The tables themselves are downloaded by RelBench; this directory only holds the
descriptions that are put into the prompt (see refuge/prompts.py:base_context).

Each rel-<dataset>/ folder contains:

  <dataset>_schema.txt   database description and every table's columns, with
                         column types and primary/foreign keys
  feature_<dataset>.txt  per-table column statistics (NA ratios, mean ± std)
  task-<task>.txt        natural-language description of the prediction task

Which files a task uses is set in refuge/tasks.py. rel-f1, rel-event, rel-avito
and rel-amazon have tasks registered there; rel-hm, rel-stack and rel-trial are
provided for adding new tasks.

Pass --data_dir to run.py to use a different directory with the same layout.
