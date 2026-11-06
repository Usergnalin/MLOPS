"""Upload cleaned customers / orders / labels CSVs from data/clean/ into the team's MySQL database.

Reads the files written by data_clean.py (data/clean/<table>_<postfix>.csv) and upserts them into
three tables that mirror the files. No cleaning is repeated here: each CSV is staged as text and
MySQL casts the values to proper types on insert, so anything that slipped past cleaning fails
loudly instead of being silently coerced.

Re-running a batch is safe (rows are upserted on their primary key). A later batch overwrites a
customer's snapshot and can update an order's label; batch_id records which batch last wrote a row.

Connection settings come from the DB_* variables in .env (see config.py). DB_NAME is created
if it doesn't exist.

Usage:
    python src/data_upload.py                      # uploads data/clean/*_base.csv
    python src/data_upload.py --postfix batch2     # uploads data/clean/*_batch2.csv
    python src/data_upload.py --reset              # drop and recreate the tables first
"""
import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from tqdm import tqdm

from config import CLEAN_DIR, db_name, make_engine, setup_logging
# data_clean is the single source of truth for columns and formats.
from data_clean import COLUMNS, FIRST_ORDER_OUT, ORDER_DATE_OUT, TABLE_NAMES

log = logging.getLogger("data_upload")

STAGING_PREFIX = "stg_"


def mysql_fmt(py_fmt):
    """strftime format -> MySQL STR_TO_DATE format (for these formats only minutes differ)."""
    return py_fmt.replace("%M", "%i")


# ---------------------------------------------------------------------------
# Table design
# Each column: (MySQL type, SELECT expression that casts the staged text column `s.<col>`).
# Column order must match COLUMNS in data_clean.py (checked at runtime).
# ---------------------------------------------------------------------------
SCHEMA = {
    "customers": {
        "columns": {
            "country_code": ("CHAR(2) NOT NULL", "s.country_code"),
            "customer_id": ("CHAR(10) NOT NULL", "s.customer_id"),
            "mobile_verified": ("BOOLEAN NOT NULL", "(s.mobile_verified = 'True')"),
            "num_orders_last_50days": ("INT NOT NULL", "CAST(s.num_orders_last_50days AS SIGNED)"),
            "num_cancelled_orders_last_50days": (
                "INT NOT NULL", "CAST(s.num_cancelled_orders_last_50days AS SIGNED)"),
            "num_refund_orders_last_50days": (
                "INT NOT NULL", "CAST(s.num_refund_orders_last_50days AS SIGNED)"),
            "total_payment_last_50days": (
                "DECIMAL(18,6) NOT NULL", "CAST(s.total_payment_last_50days AS DECIMAL(18,6))"),
            "num_associated_customers": ("INT NOT NULL", "CAST(s.num_associated_customers AS SIGNED)"),
            # Blank in the clean CSV means "unknown" -> NULL.
            "first_order_datetime": (
                "DATETIME NULL", "STR_TO_DATE(NULLIF(s.first_order_datetime, ''), :datetime_fmt)"),
        },
        # data_clean de-duplicates customers on customer_id alone, so it is globally unique.
        "primary_key": ["customer_id"],
        "extra": [],
    },
    "orders": {
        "columns": {
            "country_code": ("CHAR(2) NOT NULL", "s.country_code"),
            "order_id": ("CHAR(10) NOT NULL", "s.order_id"),
            "collect_type": ("VARCHAR(20) NOT NULL", "s.collect_type"),
            "payment_method": ("VARCHAR(100) NOT NULL", "s.payment_method"),
            "order_value": ("DECIMAL(18,6) NOT NULL", "CAST(s.order_value AS DECIMAL(18,6))"),
            "num_items_ordered": ("INT NOT NULL", "CAST(s.num_items_ordered AS SIGNED)"),
            "refund_value": ("DECIMAL(18,6) NOT NULL", "CAST(s.refund_value AS DECIMAL(18,6))"),
            "order_date": ("DATE NOT NULL", "STR_TO_DATE(s.order_date, :date_fmt)"),
        },
        # order_id is reused across countries.
        "primary_key": ["country_code", "order_id"],
        "extra": ["INDEX `idx_orders_order_date` (`order_date`)"],
    },
    "labels": {
        "columns": {
            "country_code": ("CHAR(2) NOT NULL", "s.country_code"),
            "order_id": ("CHAR(10) NOT NULL", "s.order_id"),
            "customer_id": ("CHAR(10) NOT NULL", "s.customer_id"),
            "is_fraud": ("TINYINT(1) NOT NULL", "CAST(s.is_fraud AS UNSIGNED)"),
        },
        "primary_key": ["country_code", "order_id"],
        "extra": [
            "INDEX `idx_labels_customer` (`customer_id`)",
            # data_clean guarantees every label points to an order and a customer in the same batch.
            "CONSTRAINT `fk_labels_order` FOREIGN KEY (`country_code`, `order_id`) "
            "REFERENCES `orders` (`country_code`, `order_id`)",
            "CONSTRAINT `fk_labels_customer` FOREIGN KEY (`customer_id`) "
            "REFERENCES `customers` (`customer_id`)",
        ],
    },
}


def check_schema_matches_clean():
    for t in TABLE_NAMES:
        if list(SCHEMA[t]["columns"]) != COLUMNS[t]:
            raise RuntimeError(f"SCHEMA['{t}'] columns are out of sync with data_clean.COLUMNS['{t}']")


def q(name):
    return f"`{name}`"


def create_table_sql(table):
    spec = SCHEMA[table]
    lines = [f"{q(c)} {ddl}" for c, (ddl, _) in spec["columns"].items()]
    lines += [
        "`batch_id` VARCHAR(64) NOT NULL",
        "`loaded_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP",
        "PRIMARY KEY (" + ", ".join(q(c) for c in spec["primary_key"]) + ")",
        *spec["extra"],
    ]
    return (f"CREATE TABLE IF NOT EXISTS {q(table)} (\n  " + ",\n  ".join(lines)
            + "\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4")


def upsert_sql(table):
    """INSERT ... SELECT from the staging table, casting types; update non-key columns on conflict."""
    spec = SCHEMA[table]
    cols = list(spec["columns"]) + ["batch_id"]
    exprs = [expr for _, expr in spec["columns"].values()] + [":batch_id"]
    select = ",\n    ".join(f"{e} AS {q(c)}" for c, e in zip(cols, exprs))
    updates = ",\n  ".join(f"{q(c)} = new.{q(c)}" for c in cols if c not in spec["primary_key"])
    return (
        f"INSERT INTO {q(table)} ({', '.join(q(c) for c in cols)})\n"
        f"SELECT * FROM (\n  SELECT\n    {select}\n  FROM {q(STAGING_PREFIX + table)} AS s\n) AS new\n"
        f"ON DUPLICATE KEY UPDATE\n  {updates}"
    )


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
def get_engine():
    """Engine for DB_NAME, creating the database on first use."""
    server = make_engine(with_database=False)
    try:
        with server.begin() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {q(db_name())} CHARACTER SET utf8mb4"))
    except SQLAlchemyError as e:  # e.g. user lacks CREATE privilege but the database already exists
        log.warning("could not ensure database exists (%s); continuing", e.__class__.__name__)
    finally:
        server.dispose()
    return make_engine()


def ensure_tables(engine, reset=False):
    with engine.begin() as conn:
        if reset:
            for t in reversed(TABLE_NAMES):  # children first because of foreign keys
                conn.execute(text(f"DROP TABLE IF EXISTS {q(t)}"))
            log.warning("dropped tables %s", ", ".join(TABLE_NAMES))
        for t in TABLE_NAMES:  # parents first
            conn.execute(text(create_table_sql(t)))


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def check_clean_file(path, table):
    """Only the header is read here; the rows go straight from the file to MySQL."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run data_clean.py for this postfix first")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if header != COLUMNS[table]:
        raise ValueError(f"{path.name}: columns {header} do not match {COLUMNS[table]}")


def count_rows(conn, table):
    return conn.execute(text(f"SELECT COUNT(*) FROM {q(table)}")).scalar_one()


def create_staging_table(conn, table):
    stg = q(STAGING_PREFIX + table)
    conn.execute(text(f"DROP TABLE IF EXISTS {stg}"))
    cols = ", ".join(f"{q(c)} TEXT" for c in COLUMNS[table])
    conn.execute(text(f"CREATE TABLE {stg} ({cols}) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"))


def line_ending(path):
    """pandas writes \\r\\n on Windows and \\n elsewhere, so detect it from the header line."""
    with open(path, "rb") as f:
        return "\r\n" if f.readline().endswith(b"\r\n") else "\n"


def stage(engine, table, path):
    """MySQL reads the CSV file itself (LOAD DATA LOCAL INFILE) into a TEXT staging table.

    There is deliberately no fallback: if the server refuses, the upload stops with a clear error.
    """
    cols = ", ".join(q(c) for c in COLUMNS[table])
    with engine.begin() as conn:
        create_staging_table(conn, table)
        try:
            # ESCAPED BY '' keeps backslashes literal; blank fields stay '' (as written by data_clean).
            conn.execute(text(
                f"LOAD DATA LOCAL INFILE :path INTO TABLE {q(STAGING_PREFIX + table)} "
                "CHARACTER SET utf8mb4 "
                "FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\"' ESCAPED BY '' "
                f"LINES TERMINATED BY :eol IGNORE 1 LINES ({cols})"
            ), {"path": str(Path(path).resolve()), "eol": line_ending(path)})
        except SQLAlchemyError as e:
            raise RuntimeError(
                f"LOAD DATA LOCAL INFILE failed for {Path(path).name}: {getattr(e, 'orig', e)}. "
                "If the server disallows local infile, enable local_infile on the MySQL server "
                "(on RDS: set local_infile=1 in the parameter group and reboot the instance)."
            ) from e


def run(postfix="base", clean_dir=CLEAN_DIR, reset=False, engine=None):
    """Upload one cleaned batch. Returns {table: {"rows", "inserted", "updated"}}."""
    start = time.perf_counter()
    check_schema_matches_clean()
    clean_dir = Path(clean_dir)
    paths = {t: clean_dir / f"{t}_{postfix}.csv" for t in TABLE_NAMES}
    for t, path in paths.items():
        check_clean_file(path, t)

    own_engine = engine is None
    engine = engine or get_engine()
    params = {"batch_id": postfix,
              "datetime_fmt": mysql_fmt(FIRST_ORDER_OUT),
              "date_fmt": mysql_fmt(ORDER_DATE_OUT)}
    stats, staged = {}, []
    try:
        ensure_tables(engine, reset)

        # Staging uses DDL (implicit commit in MySQL), so it happens before the transaction.
        # LOAD DATA and the upserts report no progress of their own, so the bars count tables.
        for t in tqdm(TABLE_NAMES, desc="stage (LOAD DATA)", unit="table"):
            staged.append(STAGING_PREFIX + t)
            t0 = time.perf_counter()
            stage(engine, t, paths[t])
            log.info("staged    %-9s in %6.1fs", t, time.perf_counter() - t0)

        # All three upserts commit together or not at all; parents before children for FKs.
        with engine.begin() as conn:
            for t in tqdm(TABLE_NAMES, desc="upsert", unit="table"):
                t0 = time.perf_counter()
                sql = upsert_sql(t)
                rows = count_rows(conn, STAGING_PREFIX + t)
                before = count_rows(conn, t)
                conn.execute(text(sql), {k: v for k, v in params.items() if f":{k}" in sql})
                inserted = count_rows(conn, t) - before
                stats[t] = {"rows": rows, "inserted": inserted, "updated": rows - inserted}
                log.info("upserted  %-9s in %6.1fs", t, time.perf_counter() - t0)
    finally:
        try:
            with engine.begin() as conn:
                for stg in staged:
                    conn.execute(text(f"DROP TABLE IF EXISTS {q(stg)}"))
        except SQLAlchemyError:
            log.warning("could not drop staging tables %s", staged)
        if own_engine:
            engine.dispose()

    for t, s in stats.items():
        log.info("%-9s rows=%9d  inserted=%9d  updated=%9d", t, s["rows"], s["inserted"], s["updated"])
    log.info("uploaded postfix '%s' to %s in %.1fs", postfix, db_name(), time.perf_counter() - start)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--postfix", default="base", help="batch postfix, e.g. base -> customers_base.csv")
    parser.add_argument("--clean-dir", type=Path, default=CLEAN_DIR,
                        help="folder with cleaned CSVs (default: %(default)s)")
    parser.add_argument("--reset", action="store_true", help="drop and recreate the tables before loading")
    args = parser.parse_args()
    setup_logging()
    run(args.postfix, args.clean_dir, args.reset)