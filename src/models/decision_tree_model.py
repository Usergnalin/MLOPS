"""Decision tree fraud classifier.

Reads the snapshot written by data_fetch.py, trains one sklearn Pipeline (minimal preprocessing +
DecisionTreeClassifier) and logs params, metrics and the model to the MLflow server.
Because preprocessing is inside the Pipeline, the logged model takes raw snapshot columns.

Env vars (the project .env is loaded via data_fetch):
    MLFLOW_TRACKING_URI      required, e.g. http://<your-eb-env>.elasticbeanstalk.com
    MLFLOW_EXPERIMENT_NAME   optional, default "sp-buy-fraud"
    PIPELINE_RUN_ID          optional; set by pipeline.py so every run of one ETL run shares a prefix
    MODEL_AUTHOR             optional, default: your OS user name

Usage:
    python src/models/decision_tree_model.py
    python src/models/decision_tree_model.py --max-depth 12 --min-samples-leaf 20
"""
import argparse
import getpass
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for data_fetch
from data_fetch import TARGET, load_training_data  # noqa: E402  (also loads .env)

log = logging.getLogger("decision_tree_model")

MODEL_TYPE = "decision_tree"
RANDOM_STATE = 42

CATEGORICAL = ["country_code", "collect_type", "payment_method"]
NUMERIC = [
    "order_value", "num_items_ordered", "refund_value", "mobile_verified",
    "num_orders_last_50days", "num_cancelled_orders_last_50days", "num_refund_orders_last_50days",
    "total_payment_last_50days", "num_associated_customers",
    "account_age_days",  # created by add_features
]
# Raw snapshot columns the model takes as input (IDs and batch_id are deliberately left out).
INPUT_COLUMNS = CATEGORICAL + [c for c in NUMERIC if c != "account_age_days"] + [
    "order_date", "first_order_datetime",
]


def add_features(X):
    """Minimal feature engineering, kept inside the Pipeline so scoring uses the same code."""
    X = X.copy()
    # How old the account was when the order was placed; -1 when first_order_datetime is unknown
    # (a tree can split that value off on its own).
    X["account_age_days"] = (X["order_date"] - X["first_order_datetime"]).dt.days.fillna(-1)
    X["mobile_verified"] = X["mobile_verified"].astype(int)
    return X


def build_model(max_depth, min_samples_leaf):
    preprocess = ColumnTransformer(
        [
            # handle_unknown="ignore": a payment method first seen in a new batch won't crash scoring.
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", "passthrough", NUMERIC),  # trees don't need scaling
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",  # fraud is the minority class
        random_state=RANDOM_STATE,
    )
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("preprocess", preprocess),
        ("model", tree),
    ])


def evaluate(y_true, y_pred, y_proba, prefix):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}_roc_auc": roc_auc_score(y_true, y_proba),
        f"{prefix}_pr_auc": average_precision_score(y_true, y_proba),
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_tp": int(tp), f"{prefix}_fp": int(fp),
        f"{prefix}_fn": int(fn), f"{prefix}_tn": int(tn),
    }


def run(max_depth=8, min_samples_leaf=50):
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise RuntimeError("MLFLOW_TRACKING_URI is not set (see .env.example)")

    # The train/test split is fixed by data_fetch, so every model is scored on the same rows.
    train, test, manifest = load_training_data()
    X_train, y_train = train[INPUT_COLUMNS], train[TARGET]
    X_test, y_test = test[INPUT_COLUMNS], test[TARGET]

    pipeline_run_id = os.environ.get("PIPELINE_RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S")
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "sp-buy-fraud"))

    with mlflow.start_run(run_name=f"{pipeline_run_id}_{MODEL_TYPE}") as run:
        mlflow.set_tags({
            "model_type": MODEL_TYPE,
            "author": os.environ.get("MODEL_AUTHOR") or getpass.getuser(),
            "pipeline_run_id": pipeline_run_id,
            "data_hash": manifest["data_hash"],
            "data_batches": ",".join(manifest["batches"]),
        })
        mlflow.log_params({
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "split_test_percent": manifest["split"]["test_percent"],
            "train_rows": manifest["train"]["rows"],
            "test_rows": manifest["test"]["rows"],
            "train_fraud_rate": manifest["train"]["fraud_rate"],
            "test_fraud_rate": manifest["test"]["fraud_rate"],
            "categorical_features": ",".join(CATEGORICAL),
            "numeric_features": ",".join(NUMERIC),
        })

        model = build_model(max_depth, min_samples_leaf)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        mlflow.log_metric("train_seconds", time.perf_counter() - start)

        # Train metrics alongside test metrics make over/underfitting visible in the MLflow UI.
        metrics = evaluate(y_train, model.predict(X_train), model.predict_proba(X_train)[:, 1], "train")
        test_pred = model.predict(X_test)
        metrics |= evaluate(y_test, test_pred, model.predict_proba(X_test)[:, 1], "test")
        mlflow.log_metrics(metrics)

        names = model.named_steps["preprocess"].get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_
        mlflow.log_dict(
            {n: round(float(v), 6) for n, v in sorted(zip(names, importances), key=lambda p: -p[1])},
            "feature_importances.json",
        )

        mlflow.sklearn.log_model(
            model, "model",
            signature=infer_signature(X_test.head(100), test_pred[:100]),
            input_example=X_test.head(5),
        )

    log.info("run %s  test: f1=%.4f precision=%.4f recall=%.4f roc_auc=%.4f pr_auc=%.4f",
             run.info.run_id, metrics["test_f1"], metrics["test_precision"],
             metrics["test_recall"], metrics["test_roc_auc"], metrics["test_pr_auc"])
    return run.info.run_id, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.max_depth, args.min_samples_leaf)