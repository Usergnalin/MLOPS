"""Shared settings for every script: project paths, external connections and logging.

The project .env holds only what is needed to reach external resources:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD   MySQL database (DB_PORT optional, default 3306)
    MLFLOW_TRACKING_URI                               MLflow tracking server
Variables already set in the shell take precedence over .env.

Local paths are command-line arguments; their defaults all live under data/:
    data/raw/        raw CSVs as downloaded (<table>_<postfix>.csv)
    data/clean/      output of data_clean.py
    data/training/   train / test snapshot written by data_fetch.py
    data/eda/        output of eda.ipynb
"""
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
TRAINING_DIR = DATA_DIR / "training"
EDA_DIR = DATA_DIR / "eda"

load_dotenv(ROOT / ".env")


def require_env(*names):
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise RuntimeError(f"missing environment variables: {', '.join(missing)} (see .env.example)")
    return [os.environ[n] for n in names]


# ---------------------------------------------------------------------------
# MySQL
# ---------------------------------------------------------------------------
def db_name():
    (name,) = require_env("DB_NAME")
    if not re.fullmatch(r"\w+", name):  # it is interpolated into CREATE DATABASE
        raise RuntimeError(f"DB_NAME must be letters, digits or underscores, got {name!r}")
    return name


def db_url(with_database=True):
    host, user, password = require_env("DB_HOST", "DB_USER", "DB_PASSWORD")
    return URL.create(
        "mysql+pymysql",
        username=user,
        password=password,
        host=host,
        port=int(os.environ.get("DB_PORT") or 3306),
        database=db_name() if with_database else None,
    )


def make_engine(with_database=True):
    return create_engine(db_url(with_database), pool_pre_ping=True,
                         connect_args={"connect_timeout": 10, "charset": "utf8mb4",
                                       "local_infile": True})  # allows LOAD DATA LOCAL INFILE


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
def mlflow_tracking_uri():
    return require_env("MLFLOW_TRACKING_URI")[0]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
class _TqdmHandler(logging.Handler):
    """Writes log lines through tqdm so they don't break progress bars."""

    def emit(self, record):
        tqdm.write(self.format(record))


def setup_logging(level=logging.INFO):
    handler = _TqdmHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    logging.basicConfig(level=level, handlers=[handler], force=True)
