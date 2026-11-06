"""End-to-end pipeline: clean a raw batch, upload it, fetch the training snapshot, train the models.

Steps (all run by default, always in this order):
    clean    data/raw/<table>_<postfix>.csv  -> data/clean/        (data_clean.py)
    upload   data/clean/                     -> MySQL               (data_upload.py)
    fetch    MySQL (all batches)             -> data/training/      (data_fetch.py)
    train    data/training/                  -> MLflow, per model   (models/*.py)

Every model of one pipeline run gets the same run id, so their MLflow runs group together.

Usage:
    python src/pipeline.py                            # base batch, all steps
    python src/pipeline.py --postfix batch2           # add a new batch, retrain on all batches
    python src/pipeline.py --steps fetch train        # retrain on what is already in MySQL
    python src/pipeline.py --steps train --models decision_tree
"""
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import data_clean
import data_fetch
import data_upload
from config import (
    CLEAN_DIR, DEFAULT_EXPERIMENT, RAW_DIR, TRAINING_DIR, db_url, mlflow_tracking_uri, setup_logging,
)
from models import decision_tree_model

log = logging.getLogger("pipeline")

STEPS = ["clean", "upload", "fetch", "train"]
# name -> module with run(training_dir=, experiment=, pipeline_run_id=, author=) -> (mlflow_run_id, metrics)
MODELS = {
    decision_tree_model.MODEL_TYPE: decision_tree_model,
}


def check_config(steps):
    """Fail before any long step starts if a needed connection setting is missing."""
    if {"upload", "fetch"} & set(steps):
        db_url()
    if "train" in steps:
        mlflow_tracking_uri()


def run(steps=STEPS, postfix="base", raw_dir=RAW_DIR, clean_dir=CLEAN_DIR, training_dir=TRAINING_DIR,
        reset=False, models=tuple(MODELS), experiment=DEFAULT_EXPERIMENT,
        pipeline_run_id=None, author=None):
    """Run the selected steps. Returns {model name: (mlflow_run_id, metrics)} for trained models."""
    steps = [s for s in STEPS if s in steps]
    check_config(steps)
    pipeline_run_id = pipeline_run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    log.info("pipeline run %s: steps=%s postfix=%s", pipeline_run_id, ",".join(steps), postfix)

    start = time.perf_counter()
    results = {}
    for i, step in enumerate(steps, 1):
        log.info("=== step %d/%d: %s ===", i, len(steps), step)
        t0 = time.perf_counter()
        if step == "clean":
            data_clean.run(postfix, raw_dir, clean_dir)
        elif step == "upload":
            data_upload.run(postfix, clean_dir, reset)
        elif step == "fetch":
            data_fetch.run(training_dir)
        elif step == "train":
            for name in models:
                results[name] = MODELS[name].run(training_dir=training_dir, experiment=experiment,
                                                 pipeline_run_id=pipeline_run_id, author=author)
        log.info("step %s done in %.1fs", step, time.perf_counter() - t0)

    for name, (mlflow_run_id, metrics) in results.items():
        log.info("%-15s mlflow run %s  test_f1=%.4f  test_pr_auc=%.4f",
                 name, mlflow_run_id, metrics["test_f1"], metrics["test_pr_auc"])
    log.info("pipeline run %s finished in %.1fs", pipeline_run_id, time.perf_counter() - start)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS, help="steps to run (default: all)")
    parser.add_argument("--postfix", default="base", help="batch postfix, e.g. base -> customers_base.csv")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="folder with raw CSVs (default: %(default)s)")
    parser.add_argument("--clean-dir", type=Path, default=CLEAN_DIR,
                        help="folder for cleaned CSVs (default: %(default)s)")
    parser.add_argument("--training-dir", type=Path, default=TRAINING_DIR,
                        help="folder for the train / test snapshot (default: %(default)s)")
    parser.add_argument("--reset", action="store_true", help="upload: drop and recreate the tables first")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS),
                        help="models to train (default: all)")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT,
                        help="MLflow experiment name (default: %(default)s)")
    parser.add_argument("--run-id", help="pipeline run id, used as MLflow run name prefix (default: timestamp)")
    parser.add_argument("--author", help="author tag on MLflow runs (default: OS user name)")
    args = parser.parse_args()
    setup_logging()
    run(args.steps, args.postfix, args.raw_dir, args.clean_dir, args.training_dir, args.reset,
        args.models, args.experiment, args.run_id, args.author)
