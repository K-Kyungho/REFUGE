"""Per-task configuration.

Everything that differed between the original per-task scripts lives here;
the pipeline itself is shared.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

SONNET_4 = "claude-sonnet-4-20250514"


@dataclass(frozen=True)
class TaskConfig:
    dataset: str                      # RelBench dataset name, e.g. "rel-f1"
    task: str                         # RelBench task name, e.g. "driver-dnf"
    entity_table: str                 # table the features are attached to
    entity_col: str                   # primary key of entity_table
    target_col: str                   # label column in the task table
    time_col: str                     # timestamp column in the task table
    schema_file: str                  # paths relative to data/
    stats_file: str
    task_file: str
    drop_cols: Tuple[str, ...]        # excluded from the LightGBM evaluator
    categorical_cols: Tuple[str, ...] = ()
    # The RelBench task table names the entity key differently from the
    # entity table (rel-event: "user" vs "user_id"); rename it on load.
    task_col_rename: Dict[str, str] = field(default_factory=dict)
    planner_models: Tuple[str, str, str] = (SONNET_4, SONNET_4, SONNET_4)


_F1 = dict(
    dataset="rel-f1", entity_table="drivers", entity_col="driverId", time_col="date",
    schema_file="rel-f1/f1_schema.txt", stats_file="rel-f1/feature_f1.txt",
    drop_cols=("date", "driverRef", "code", "forename", "surname", "dob"),
    categorical_cols=("nationality",),
)
_EVENT = dict(
    dataset="rel-event", entity_table="users", entity_col="user_id", time_col="timestamp",
    target_col="target",
    schema_file="rel-event/event_schema.txt", stats_file="rel-event/feature_event.txt",
    drop_cols=("index", "timestamp", "user_id", "joinedAt", "location"),
    categorical_cols=("locale", "gender"),
    task_col_rename={"user": "user_id"},
)
_AVITO = dict(
    dataset="rel-avito", entity_table="UserInfo", entity_col="UserID", time_col="timestamp",
    target_col="num_click",
    schema_file="rel-avito/avito_schema.txt", stats_file="rel-avito/feature_avito.txt",
    drop_cols=("UserID", "timestamp"),
    planner_models=(SONNET_4, "claude-haiku-4-5-20251001", "claude-3-7-sonnet-20250219"),
)

TASKS: Dict[str, TaskConfig] = {
    "f1-dnf": TaskConfig(**_F1, task="driver-dnf", target_col="did_not_finish",
                         task_file="rel-f1/task-driver-dnf.txt"),
    "f1-top3": TaskConfig(**_F1, task="driver-top3", target_col="qualifying",
                          task_file="rel-f1/task-driver-top3.txt"),
    "event-repeat": TaskConfig(**_EVENT, task="user-repeat",
                               task_file="rel-event/task-user-repeat.txt"),
    "event-ignore": TaskConfig(**_EVENT, task="user-ignore",
                               task_file="rel-event/task-user-ignore.txt"),
    "avito-clicks": TaskConfig(**_AVITO, task="user-clicks",
                               task_file="rel-avito/task-user-clicks.txt"),
    "avito-visits": TaskConfig(**_AVITO, task="user-visits",
                               task_file="rel-avito/task-user-visits.txt"),
    "amazon-churn": TaskConfig(
        dataset="rel-amazon", task="user-churn", entity_table="customer",
        entity_col="customer_id", target_col="churn", time_col="timestamp",
        schema_file="rel-amazon/amazon_schema.txt", stats_file="rel-amazon/feature_amazon.txt",
        task_file="rel-amazon/task-user-churn.txt",
        drop_cols=("timestamp", "customer_name", "customer_id"),
    ),
}
