"""
Regression Models Module.
Implements:
- Experiment 1: Linear Regression & Polynomial Regression
- Experiment 3: K-Nearest Neighbors (KNN) Regressor with K-search
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

from src.evaluation import evaluate_regression


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_cols: List[str]
) -> Dict[str, Any]:
    """
    Trains standard Multiple Linear Regression to predict later assessment performance.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", LinearRegression())
    ])

    pipeline.fit(X_train[numeric_cols], y_train)
    y_pred = pipeline.predict(X_test[numeric_cols])

    metrics = evaluate_regression(y_test, y_pred)

    # Feature coefficients
    reg_step = pipeline.named_steps["reg"]
    coefs = reg_step.coef_
    coef_df = pd.DataFrame({
        "Feature": numeric_cols,
        "Coefficient": coefs,
        "Abs_Coefficient": np.abs(coefs)
    }).sort_values(by="Abs_Coefficient", ascending=False).reset_index(drop=True)

    residuals = y_test.values - y_pred

    return {
        "model": pipeline,
        "metrics": metrics,
        "y_pred": y_pred,
        "residuals": residuals,
        "coefficients": coef_df
    }


def train_polynomial_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_cols: List[str],
    degree: int = 2
) -> Dict[str, Any]:
    """
    Trains Polynomial Regression using PolynomialFeatures and Ridge/Linear regression
    to model nonlinear relationships and feature interactions.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("reg", Ridge(alpha=10.0))  # Ridge regularizer to prevent overfitting on polynomial features
    ])

    pipeline.fit(X_train[numeric_cols], y_train)
    y_pred = pipeline.predict(X_test[numeric_cols])

    metrics = evaluate_regression(y_test, y_pred)
    residuals = y_test.values - y_pred

    poly_step = pipeline.named_steps["poly"]
    poly_features_count = len(poly_step.get_feature_names_out(numeric_cols))

    return {
        "model": pipeline,
        "metrics": metrics,
        "y_pred": y_pred,
        "residuals": residuals,
        "degree": degree,
        "poly_features_count": poly_features_count
    }


def train_knn_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_cols: List[str],
    k: int = 5,
    weights: str = "distance"
) -> Dict[str, Any]:
    """
    Trains KNN Regressor with safety check on k <= len(X_train).
    """
    safe_k = max(1, min(k, len(X_train)))
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", KNeighborsRegressor(n_neighbors=safe_k, weights=weights))
    ])

    pipeline.fit(X_train[numeric_cols], y_train)
    y_pred = pipeline.predict(X_test[numeric_cols])

    metrics = evaluate_regression(y_test, y_pred)
    residuals = y_test.values - y_pred

    return {
        "model": pipeline,
        "metrics": metrics,
        "y_pred": y_pred,
        "residuals": residuals,
        "k": safe_k
    }


def run_knn_regressor_k_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_cols: List[str],
    k_range: range = range(1, 21),
    max_train_samples: int = 5000
) -> pd.DataFrame:
    """
    Compares KNN Regressor across K values (1 to 20) and records MAE, MSE, RMSE, R2.
    """
    if len(X_train) > max_train_samples:
        indices = np.random.RandomState(42).choice(len(X_train), max_train_samples, replace=False)
        X_tr = X_train.iloc[indices][numeric_cols]
        y_tr = y_train.iloc[indices]
    else:
        X_tr = X_train[numeric_cols]
        y_tr = y_train

    # Pre-scale once for fast execution
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr_trans = scaler.fit_transform(imputer.fit_transform(X_tr))
    X_te_trans = scaler.transform(imputer.transform(X_test[numeric_cols]))

    records = []
    max_k = min(len(X_tr), max(k_range))
    actual_k_range = [k for k in k_range if 1 <= k <= max_k]

    for k in actual_k_range:
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
        knn.fit(X_tr_trans, y_tr)
        y_pred = knn.predict(X_te_trans)

        m = evaluate_regression(y_test, y_pred)
        records.append({
            "K": k,
            "RMSE": m["rmse"],
            "MAE": m["mae"],
            "MSE": m["mse"],
            "R2": m["r2"]
        })

    return pd.DataFrame(records)
