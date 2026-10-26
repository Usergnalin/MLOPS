"""Clean raw customers / orders / labels CSVs based on the findings in eda.ipynb.

Output keeps the raw format: same files, same columns, same column order, same value formats.
Only bad values are fixed; rows that cannot be fixed go to a rejects file with a reason.
Outliers and business-rule oddities (e.g. refund > order value) are NOT touched here,
they are left for the transform / feature-engineering step.

Usage:
    python src/data_clean.py                      # cleans data/*_base.csv -> clean_data/
    python src/data_clean.py --postfix batch2     # cleans data/*_batch2.csv
    python src/data_clean.py --data-dir X --out-dir Y
"""
import argparse
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("data_clean")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
CLEAN_DATA_DIR = Path(os.environ.get("CLEAN_DATA_DIR", ROOT / "clean_data"))

# ---------------------------------------------------------------------------
# Config (mirrors the Config cell in eda.ipynb)
# ---------------------------------------------------------------------------
TABLE_NAMES = ["customers", "orders", "labels"]

COLUMNS = {
    "customers": [
        "country_code", "customer_id", "mobile_verified", "num_orders_last_50days",
        "num_cancelled_orders_last_50days", "num_refund_orders_last_50days",
        "total_payment_last_50days", "num_associated_customers", "first_order_datetime",
    ],
    "orders": [
        "country_code", "order_id", "collect_type", "payment_method",
        "order_value", "num_items_ordered", "refund_value", "order_date",
    ],
    "labels": ["country_code", "order_id", "customer_id", "is_fraud"],
}

# order_id is reused across countries, so orders/labels are keyed on (country_code, order_id).
ORDER_KEY = ["country_code", "order_id"]

NULL_TOKENS = {"", "nan", "null", "none", "n/a", "na", "-", "?", "nat", "missing"}
COUNTRIES = {"BD", "MY", "PH", "PK", "TH"}
ID_PATTERN = re.compile(r"^[0-9a-f]{10}$")

BOOL_MAP = {"true": "True", "yes": "True", "y": "True", "1": "True",
            "false": "False", "no": "False", "n": "False", "0": "False"}

# Canonical spelling = the most frequent raw spelling in the base data.
COLLECT_TYPE_MAP = {"delivery": "Delivery", "pickup": "pickup"}

PAYMENT_METHODS = [
    "payment on Delivery", "cybersource", "online banking", "credit card", "antfinancial gcash",
    "generic creditcard", "balance", "antfinancial bkash", "antfinancial truemoney", "boost wallet",
    "jazzcash wallet", "invoice", "adyen hpp molpay", "no payment", "cybersource applepay",
    "paypal", "xendit directdebit",
]
PAYMENT_METHOD_MAP = {m.lower(): m for m in PAYMENT_METHODS}
PAYMENT_METHOD_MAP["on delivery"] = "payment on Delivery"  # synonym found in EDA
PAYMENT_METHOD_UNKNOWN = "unknown"  # all blanks are in MY; kept as an explicit category
PAYMENT_METHOD_MAP[PAYMENT_METHOD_UNKNOWN] = PAYMENT_METHOD_UNKNOWN

INT_COLUMNS = {
    "customers": {"num_orders_last_50days": 0, "num_cancelled_orders_last_50days": 0,
                  "num_refund_orders_last_50days": 0, "num_associated_customers": 1},
    "orders": {"num_items_ordered": 1},
}
FLOAT_COLUMNS = {
    "customers": {"total_payment_last_50days": 0},
    "orders": {"order_value": 0, "refund_value": 0},
}

# Expected format first; the rest are formats seen in the data. No free inference (day/month swaps).
ORDER_DATE_OUT = "%Y/%m/%d"
ORDER_DATE_FORMATS = [ORDER_DATE_OUT, "%Y-%m-%d", "%Y/%b/%d", "%Y-%b-%d"]
FIRST_ORDER_OUT = "%Y-%m-%d %H:%M:%S"
FIRST_ORDER_FORMATS = [FIRST_ORDER_OUT, "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class Cleaner:
    """Holds one table while it is cleaned; collects fix counts and rejected rows."""

    def __init__(self, table, df):
        self.table = table
        self.df = df.copy()
        self.df["_row"] = np.arange(len(df)) + 2  # line number in the raw CSV (header = line 1)
        self.fixes = {}
        self.rejects = []

    def fix(self, column, new_values):
        """Replace a column, counting how many values changed."""
        changed = int((self.df[column] != new_values).sum())
        if changed:
            self.fixes[f"{column}"] = self.fixes.get(column, 0) + changed
        self.df[column] = new_values

    def reject(self, mask, reason):
        """Move rows matching mask to the rejects list."""
        mask = pd.Series(mask, index=self.df.index) if isinstance(mask, np.ndarray) else mask
        mask = mask.fillna(False).astype(bool)
        if mask.any():
            bad = self.df[mask]
            self.rejects.append(make_rejects(self.table, bad, reason))
            self.df = self.df[~mask]


def make_rejects(table, rows, reason):
    data_cols = [c for c in rows.columns if not c.startswith("_")]
    return pd.DataFrame({
        "table": table,
        "row": rows["_row"].values,
        "reason": reason,
        "raw": [json.dumps(r, ensure_ascii=False) for r in rows[data_cols].to_dict("records")],
    })


def is_missing(s):
    return s.str.strip().str.lower().isin(NULL_TOKENS)


def parse_dates(s, formats):
    """Try each format in order; returns datetimes (NaT where nothing matched)."""
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in formats:
        todo = out.isna()
        if not todo.any():
            break
        out[todo] = pd.to_datetime(s[todo], format=fmt, errors="coerce")
    return out


def clean_country(c):
    c.fix("country_code", c.df["country_code"].str.strip().str.upper())
    c.reject(~c.df["country_code"].isin(COUNTRIES), "country_code not in allowed list")


def clean_ids(c, columns):
    for col in columns:
        c.fix(col, c.df[col].str.strip().str.lower())
        c.reject(~c.df[col].str.match(ID_PATTERN), f"{col} bad format")


def clean_ints(c):
    for col, minimum in INT_COLUMNS.get(c.table, {}).items():
        c.reject(is_missing(c.df[col]), f"{col} missing")
        num = pd.to_numeric(c.df[col].str.strip(), errors="coerce")
        bad = num.isna() | ~np.isfinite(num) | (num != np.floor(num)) | (num < minimum)
        c.reject(bad, f"{col} not an integer >= {minimum}")
        # '4.0' -> '4'
        c.fix(col, num[~bad].astype("int64").astype(str))


def clean_floats(c):
    for col, minimum in FLOAT_COLUMNS.get(c.table, {}).items():
        c.reject(is_missing(c.df[col]), f"{col} missing")
        num = pd.to_numeric(c.df[col].str.strip(), errors="coerce")
        c.reject(num.isna() | ~np.isfinite(num) | (num < minimum), f"{col} not a number >= {minimum}")
        # Raw string kept as-is to preserve full precision.
        c.fix(col, c.df[col].str.strip())


def drop_duplicates(c, key):
    """Exact duplicates are dropped; rows sharing a key with different values are rejected."""
    data_cols = [col for col in c.df.columns if not col.startswith("_")]
    exact = c.df.duplicated(subset=data_cols)
    if exact.any():
        c.fixes["duplicate_rows_dropped"] = int(exact.sum())
        c.df = c.df[~exact]
    c.reject(c.df.duplicated(subset=key, keep=False), f"conflicting duplicate {'+'.join(key)}")


# ---------------------------------------------------------------------------
# Per-table cleaning (row-level)
# ---------------------------------------------------------------------------
def clean_customers(df):
    c = Cleaner("customers", df)
    clean_country(c)
    clean_ids(c, ["customer_id"])

    mv = c.df["mobile_verified"].str.strip().str.lower().map(BOOL_MAP)
    c.reject(mv.isna(), "mobile_verified not a boolean")
    c.fix("mobile_verified", mv.dropna())

    clean_ints(c)
    clean_floats(c)

    # Missing first_order_datetime stays blank (no imputation); other formats are normalised.
    raw = c.df["first_order_datetime"]
    missing = is_missing(raw)
    parsed = parse_dates(raw.str.strip(), FIRST_ORDER_FORMATS)
    c.reject(~missing & parsed.isna(), "first_order_datetime unparseable")
    missing, parsed = missing[c.df.index], parsed[c.df.index]
    c.fix("first_order_datetime", parsed.dt.strftime(FIRST_ORDER_OUT).where(~missing, ""))

    drop_duplicates(c, ["customer_id"])
    return c


def clean_orders(df):
    c = Cleaner("orders", df)
    clean_country(c)
    clean_ids(c, ["order_id"])

    ct = c.df["collect_type"].str.strip().str.lower().map(COLLECT_TYPE_MAP)
    c.reject(ct.isna(), "collect_type not in allowed list")
    c.fix("collect_type", ct.dropna())

    pm_raw = c.df["payment_method"].str.strip()
    pm = pm_raw.str.lower().map(PAYMENT_METHOD_MAP)
    pm = pm.where(~is_missing(pm_raw), PAYMENT_METHOD_UNKNOWN)
    new_methods = pm.isna()
    if new_methods.any():
        # Unseen method: keep it (stripped) rather than lose the order, but make it visible.
        log.warning("orders: %d rows with unseen payment_method %s", new_methods.sum(),
                    sorted(pm_raw[new_methods].unique())[:10])
    c.fix("payment_method", pm.fillna(pm_raw))

    clean_ints(c)
    clean_floats(c)

    parsed = parse_dates(c.df["order_date"].str.strip(), ORDER_DATE_FORMATS)
    c.reject(parsed.isna(), "order_date missing or unparseable")
    c.fix("order_date", parsed[c.df.index].dt.strftime(ORDER_DATE_OUT))

    drop_duplicates(c, ORDER_KEY)
    return c


def clean_labels(df):
    c = Cleaner("labels", df)
    clean_country(c)
    clean_ids(c, ["order_id", "customer_id"])

    fraud = c.df["is_fraud"].str.strip()
    fraud = fraud.where(~fraud.isin(["0.0", "1.0"]), fraud.str[0])
    c.reject(~fraud.isin(["0", "1"]), "is_fraud not 0/1")
    c.fix("is_fraud", fraud[c.df.index])

    drop_duplicates(c, ORDER_KEY)
    return c


# ---------------------------------------------------------------------------
# Cross-table cleaning
# ---------------------------------------------------------------------------
def keys_of(df, cols):
    return pd.MultiIndex.from_frame(df[cols]) if len(cols) > 1 else pd.Index(df[cols[0]])


def clean_all(customers, orders, labels):
    """Clean the three raw tables (all-string DataFrames).

    Returns (clean, rejects, fixes): clean is a dict of DataFrames in the raw column layout.
    """
    cleaners = {
        "customers": clean_customers(customers),
        "orders": clean_orders(orders),
        "labels": clean_labels(labels),
    }
    cu, od, lb = (cleaners[t] for t in TABLE_NAMES)

    # A label must point to a known order (same country) and a known customer.
    lb.reject(~keys_of(lb.df, ORDER_KEY).isin(keys_of(od.df, ORDER_KEY)),
              "order (country_code, order_id) not in orders")
    lb.reject(~lb.df["customer_id"].isin(cu.df["customer_id"]), "customer_id not in customers")
    # Every order needs a label (needed for training); customers without labels are fine.
    od.reject(~keys_of(od.df, ORDER_KEY).isin(keys_of(lb.df, ORDER_KEY)), "order has no valid label")

    clean = {t: c.df[COLUMNS[t]].reset_index(drop=True) for t, c in cleaners.items()}
    all_rejects = [r for c in cleaners.values() for r in c.rejects]
    rejects = (pd.concat(all_rejects, ignore_index=True) if all_rejects
               else pd.DataFrame(columns=["table", "row", "reason", "raw"]))
    fixes = {t: c.fixes for t, c in cleaners.items()}
    return clean, rejects, fixes


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
def read_raw(path, table):
    # Everything as strings so nothing is silently converted before cleaning.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    missing = set(COLUMNS[table]) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    return df[COLUMNS[table]]


def run(postfix="base", data_dir=DATA_DIR, out_dir=CLEAN_DATA_DIR):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    raw = {t: read_raw(data_dir / f"{t}_{postfix}.csv", t) for t in TABLE_NAMES}
    clean, rejects, fixes = clean_all(raw["customers"], raw["orders"], raw["labels"])

    out_dir.mkdir(parents=True, exist_ok=True)
    for t in TABLE_NAMES:
        clean[t].to_csv(out_dir / f"{t}_{postfix}.csv", index=False)
    rejects.to_csv(out_dir / f"rejects_{postfix}.csv", index=False)

    for t in TABLE_NAMES:
        n_rej = int((rejects["table"] == t).sum())
        log.info("%-9s raw=%9d  clean=%9d  rejected=%6d  fixes=%s",
                 t, len(raw[t]), len(clean[t]), n_rej, fixes[t])
    if len(rejects):
        log.info("reject reasons:\n%s", rejects.groupby(["table", "reason"]).size().to_string())
    log.info("wrote cleaned CSVs to %s", out_dir)
    return clean, rejects, fixes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--postfix", default="base", help="batch postfix, e.g. base -> customers_base.csv")
    parser.add_argument("--data-dir", default=DATA_DIR, help="folder with raw CSVs (env DATA_DIR)")
    parser.add_argument("--out-dir", default=CLEAN_DATA_DIR, help="folder for cleaned CSVs (env CLEAN_DATA_DIR)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.postfix, args.data_dir, args.out_dir)
