"""LightGBM validation-AUROC check used to accept/reject generated features."""

import random
from typing import Iterable

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    learning_rate=0.05,
    feature_fraction=0.8,
    max_depth=-1,
    num_leaves=10,
    verbose=-1,
)


def validation_auroc(train_df: pd.DataFrame, valid_df: pd.DataFrame, target_col: str,
                     drop_cols: Iterable[str] = (), categorical_cols: Iterable[str] = (),
                     sample_frac: float = 0.3) -> float:
    """Train LightGBM on a random `sample_frac` of train and return AUROC on valid."""
    train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")].fillna(0)
    valid_df = valid_df.loc[:, ~valid_df.columns.str.contains("^Unnamed")].fillna(0)
    train_df = train_df.sample(frac=sample_frac, random_state=random.randint(0, 10**6))

    drop = list(drop_cols) + [target_col]
    X_train = train_df.drop(columns=drop, errors="ignore")
    y_train = train_df[target_col].astype(int)
    X_val = valid_df.drop(columns=drop, errors="ignore")
    y_val = valid_df[target_col].astype(int)

    # Label-encode categoricals using train classes; unseen values -> -1.
    for col in categorical_cols:
        le = LabelEncoder().fit(X_train[col].astype(str))
        mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
        X_train[col] = X_train[col].astype(str).map(mapping).fillna(-1).astype(int)
        X_val[col] = X_val[col].astype(str).map(mapping).fillna(-1).astype(int)

    model = lgb.train(
        params=LGB_PARAMS,
        train_set=lgb.Dataset(X_train, label=y_train),
        valid_sets=[lgb.Dataset(X_val, label=y_val)],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(40)],
    )
    y_prob = model.predict(X_val, num_iteration=model.best_iteration)
    return float(roc_auc_score(y_val, y_prob))
