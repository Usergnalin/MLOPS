# ST1519 CA1 - SP-buy Fraud Detection

## Setup

1. Create a virtual environment:

   ```
   python -m venv .venv
   ```

2. Activate it:

   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - macOS/Linux: `source .venv/bin/activate`

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and fill in the MySQL and MLflow connection settings.
   `.env` only holds settings for external resources; local paths are command-line arguments.

5. Download the CSV files from BrightSpace and put them in `data/raw/`.

## Data layout

Everything under `data/` is gitignored.

| Folder | Contents | Written by |
|---|---|---|
| `data/raw/` | raw CSVs as downloaded (`<table>_<postfix>.csv`) | you |
| `data/clean/` | cleaned CSVs + `rejects_<postfix>.csv` | `src/data_clean.py` |
| `data/training/` | `train.csv`, `test.csv`, `manifest.json` | `src/data_fetch.py` |
| `data/eda/` | issue tables and profiles | `src/eda.ipynb` |

## Running

Whole pipeline (clean → upload to MySQL → fetch train/test → train and log models to MLflow):

```
python src/pipeline.py                        # base batch
python src/pipeline.py --postfix batch2       # add a new batch, retrain on all batches
python src/pipeline.py --steps fetch train    # retrain on what is already in MySQL
```

Each step can also be run on its own:

```
python src/data_clean.py --postfix base
python src/data_upload.py --postfix base
python src/data_fetch.py
python src/models/decision_tree_model.py --max-depth 12
```

Run any script with `--help` for its options.

## Adding a model

Copy `src/models/decision_tree_model.py`, keep the same `run(training_dir=, experiment=, pipeline_run_id=, author=)`
signature, and register the module in `MODELS` in `src/pipeline.py`. Feature engineering that the model's
sklearn Pipeline calls goes in `src/features.py`, so the model logged to MLflow can be loaded elsewhere.
