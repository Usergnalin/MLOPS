"""Feature engineering shared by the model pipelines.

Lives in its own module (not in a model script) because a fitted sklearn Pipeline pickles these
functions by reference. Models log this file with mlflow `code_paths`, so a model loaded from
MLflow can import `features` without the rest of the project.
"""


def add_features(X):
    """Minimal feature engineering on raw snapshot columns; returns a copy."""
    X = X.copy()
    # How old the account was when the order was placed; -1 when first_order_datetime is unknown
    # (a tree can split that value off on its own).
    X["account_age_days"] = (X["order_date"] - X["first_order_datetime"]).dt.days.fillna(-1)
    X["mobile_verified"] = X["mobile_verified"].astype(int)
    return X
