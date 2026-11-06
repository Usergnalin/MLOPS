"""Decision tree fraud classifier.

Reads the snapshot written by data_fetch.py, trains one sklearn Pipeline (minimal preprocessing +
DecisionTreeClassifier) and logs params, metrics and the model to the MLflow server
(MLFLOW_TRACKING_URI in .env). Because preprocessing is inside the Pipeline, the logged model
takes raw snapshot columns.

Usage:
    python src/models/decision_tree_model.py
    python src/models/decision_tree_model.py --max-depth 12 --min-samples-leaf 20
"""
import argparse
import getpass
import logging
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
from tqdm import tqdm

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))  # so the shared src/ modules import when run as a script
import features  # noqa: E402
from config import DEFAULT_EXPERIMENT, TRAINING_DIR, mlflow_tracking_uri, setup_logging  # noqa: E402
from data_fetch import TARGET, load_training_data  # noqa: E402

log = logging.getLogger("decision_tree_model")

MODEL_TYPE = "decision_tree"
RANDOM_STATE = 42
SKOPS_TRUSTED_TYPES = ["features.add_features", "sklearn.tree._tree.Tree"]

CATEGORICAL = ["country_code", "collect_type", "payment_method"]
NUMERIC = [
    "order_value", "num_items_ordered", "refund_value", "mobile_verified",
    "num_orders_last_50days", "num_cancelled_orders_last_50days", "num_refund_orders_last_50days",
    "total_payment_last_50days", "num_associated_customers",
    "account_age_days",  # created by features.add_features
]
# Raw snapshot columns the model takes as input (IDs and batch_id are deliberately left out).
INPUT_COLUMNS = CATEGORICAL + [c for c in NUMERIC if c != "account_age_days"] + [
    "order_date", "first_order_datetime",
]


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
        # Feature engineering inside the Pipeline, so scoring uses the same code.
        ("features", FunctionTransformer(features.add_features)),
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


def run(max_depth=8, min_samples_leaf=50, training_dir=TRAINING_DIR, experiment=DEFAULT_EXPERIMENT,
        pipeline_run_id=None, author=None):
    """Train, evaluate and log one model. Returns (mlflow_run_id, metrics).

    pipeline_run_id prefixes the MLflow run name so all models of one pipeline run group together.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri())

    # The train/test split is fixed by data_fetch, so every model is scored on the same rows.
    train, test, manifest = load_training_data(training_dir)
    X_train, y_train = train[INPUT_COLUMNS], train[TARGET]
    X_test, y_test = test[INPUT_COLUMNS], test[TARGET]

    pipeline_run_id = pipeline_run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    mlflow.set_experiment(experiment)

    steps = tqdm(total=3, desc=MODEL_TYPE, unit="step")
    with steps, mlflow.start_run(run_name=f"{pipeline_run_id}_{MODEL_TYPE}") as run:
        mlflow.set_tags({
            "model_type": MODEL_TYPE,
            "author": author or getpass.getuser(),
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
        steps.set_postfix_str("fit")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        mlflow.log_metric("train_seconds", time.perf_counter() - start)
        steps.update()

        # Train metrics alongside test metrics make over/underfitting visible in the MLflow UI.
        steps.set_postfix_str("evaluate")
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
        steps.update()

        steps.set_postfix_str("log model")
        mlflow.sklearn.log_model(
            model, name="model",
            # Inferred on all test rows so rare nulls (first_order_datetime) are marked optional.
            signature=infer_signature(X_test, test_pred),
            input_example=X_test.head(5),
            code_paths=[features.__file__],  # the Pipeline references features.add_features
            # MLflow saves sklearn models with skops, which only loads types listed here.
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        steps.update()

    log.info("run %s  test: f1=%.4f precision=%.4f recall=%.4f roc_auc=%.4f pr_auc=%.4f",
             run.info.run_id, metrics["test_f1"], metrics["test_precision"],
             metrics["test_recall"], metrics["test_roc_auc"], metrics["test_pr_auc"])
    return run.info.run_id, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    parser.add_argument("--training-dir", type=Path, default=TRAINING_DIR,
                        help="snapshot written by data_fetch.py (default: %(default)s)")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="MLflow experiment name")
    parser.add_argument("--run-id", help="pipeline run id used as MLflow run name prefix (default: timestamp)")
    parser.add_argument("--author", help="author tag (default: OS user name)")
    args = parser.parse_args()
    setup_logging()
    run(args.max_depth, args.min_samples_leaf, args.training_dir, args.experiment, args.run_id, args.author)