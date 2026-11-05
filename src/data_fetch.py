"""Fetch the joined training dataset from MySQL into a local folder for the training step.

Fetched once per pipeline run; every training script reads the same snapshot through
load_training_data(), so all models in a run are trained on identical data.

Writes to TRAIN_DATA_DIR (default <project>/train_data):
    training_data.csv   one row per labelled order, joined with its customer
    manifest.json       row count, fraud rate, batches included, file hash, fetch time

The data is ordered by (country_code, order_id), so the same database contents always give the
same file and the same data_hash, and seeded train/test splits are reproducible.

Usage:
    python src/data_fetch.py
    python src/data_fetch.py --out-dir X

In a training script:
    from data_fetch import load_training_data
    df, manifest = load_training_data()   # columns already typed (floats, ints, datetimes, bool)
"""
import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from data_upload import db_url, make_engine  # same .env and connection settings as the upload

log = logging.getLogger("data_fetch")

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATA_DIR = Path(os.environ.get("TRAIN_DATA_DIR", ROOT / "training_data"))
DATA_FILE = "training_data.csv"
MANIFEST_FILE = "manifest.json"
FETCH_CHUNKSIZE = 100_000
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

TARGET = "is_fraud"

QUERY = """
SELECT
    o.country_code,
    o.order_id,
    l.customer_id,
    o.collect_type,
    o.payment_method,
    o.order_value,
    o.num_items_ordered,
    o.refund_value,
    o.order_date,
    c.mobile_verified,
    c.num_orders_last_50days,
    c.num_cancelled_orders_last_50days,
    c.num_refund_orders_last_50days,
    c.total_payment_last_50days,
    c.num_associated_customers,
    c.first_order_datetime,
    o.batch_id,
    l.is_fraud
FROM orders AS o
JOIN labels AS l
    ON l.country_code = o.country_code AND l.order_id = o.order_id
JOIN customers AS c
    ON c.customer_id = l.customer_id
ORDER BY o.country_code, o.order_id
"""

# Column types of the snapshot. Applied after the fetch and again when a CSV is loaded,
# so training scripts always get the same dtypes.
DTYPES = {
    "country_code": "string",
    "order_id": "string",
    "customer_id": "string",
    "collect_type": "string",
    "payment_method": "string",
    "order_value": "float64",            # MySQL DECIMAL arrives as Python Decimal
    "num_items_ordered": "int64",
    "refund_value": "float64",
    "mobile_verified": "bool",
    "num_orders_last_50days": "int64",
    "num_cancelled_orders_last_50days": "int64",
    "num_refund_orders_last_50days": "int64",
    "total_payment_last_50days": "float64",
    "num_associated_customers": "int64",
    "batch_id": "string",
    TARGET: "int8",
}
DATETIME_COLUMNS = ["order_date", "first_order_datetime"]  # first_order_datetime may be NaT


def set_types(df):
    for c in DATETIME_COLUMNS:
        df[c] = pd.to_datetime(df[c])
    return df.astype(DTYPES)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(engine):
    with engine.connect().execution_options(stream_results=True) as conn:
        chunks = pd.read_sql(text(QUERY), conn, chunksize=FETCH_CHUNKSIZE)
        df = pd.concat(chunks, ignore_index=True)
    if df.empty:
        raise RuntimeError("no labelled orders in the database - run data_upload.py first")
    return set_types(df)


def build_manifest(df, digest):
    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "database": os.environ["DB_NAME"],
        "host": os.environ["DB_HOST"],
        "rows": len(df),
        "fraud_rows": int(df[TARGET].sum()),
        "fraud_rate": round(float(df[TARGET].mean()), 6),
        "batches": {str(k): int(v) for k, v in df["batch_id"].value_counts().sort_index().items()},
        "order_date_min": df["order_date"].min().date().isoformat(),
        "order_date_max": df["order_date"].max().date().isoformat(),
        "data_hash": digest,
        "columns": list(df.columns),
    }


def write_atomic(path, write):
    """Write to a temp file then rename, so a reader never sees a half-written file."""
    tmp = path.with_name(f".{path.name}.tmp")
    write(tmp)
    os.replace(tmp, path)


def run(out_dir=TRAIN_DATA_DIR, engine=None):
    start = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    own_engine = engine is None
    engine = engine or make_engine(db_url())
    try:
        df = fetch(engine)
    finally:
        if own_engine:
            engine.dispose()
    log.info("fetched %d rows in %.1fs", len(df), time.perf_counter() - start)

    data_path = out_dir / DATA_FILE
    write_atomic(data_path, lambda p: df.to_csv(p, index=False, date_format=DATETIME_FORMAT,
                                                lineterminator="\n"))
    manifest = build_manifest(df, file_hash(data_path))
    write_atomic(out_dir / MANIFEST_FILE,
                 lambda p: p.write_text(json.dumps(manifest, indent=2) + "\n"))

    log.info("rows=%d  fraud_rate=%.4f  batches=%s  hash=%s",
             manifest["rows"], manifest["fraud_rate"], manifest["batches"], manifest["data_hash"][:12])
    log.info("wrote %s in %.1fs", data_path, time.perf_counter() - start)
    return df, manifest


def load_training_data(data_dir=TRAIN_DATA_DIR):
    """For training scripts: returns (DataFrame, manifest) from the latest fetch."""
    data_dir = Path(data_dir)
    data_path, manifest_path = data_dir / DATA_FILE, data_dir / MANIFEST_FILE
    if not data_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"no training snapshot in {data_dir} - run data_fetch.py first")
    manifest = json.loads(manifest_path.read_text())
    if file_hash(data_path) != manifest["data_hash"]:
        raise RuntimeError(f"{DATA_FILE} does not match {MANIFEST_FILE} (hash differs); "
                           "re-run data_fetch.py")
    df = pd.read_csv(data_path, dtype=DTYPES, parse_dates=DATETIME_COLUMNS,
                     date_format=DATETIME_FORMAT)
    return df, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=TRAIN_DATA_DIR, help="output folder (env TRAIN_DATA_DIR)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.out_dir)