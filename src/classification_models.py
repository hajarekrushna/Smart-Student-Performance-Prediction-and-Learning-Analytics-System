"""
Classification Models Module.
Implements:
- Experiment 1: Logistic Regression
- Experiment 2: Support Vector Machine (SVM) with Kernels & 2D Decision Boundary
- Experiment 3: K-Nearest Neighbors (KNN) Classifier with K-search
- Experiment 7: Ensemble Learning with AdaBoost (Decision Stumps)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

from src.evaluation import evaluate_classification


def get_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Constructs a robust sklearn ColumnTransformer for numeric and categorical columns.
    Prevents leakage by fitting only on training data.
    """
    transformers = []
    if numeric_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, numeric_cols))

    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    C: float = 1.0,
    max_iter: int = 1000
) -> Dict[str, Any]:
    """
    Trains Logistic Regression within a pipeline.
    Returns metrics, predictions, probabilities, and model coefficients.
    """
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(C=C, max_iter=max_iter, random_state=42, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = evaluate_classification(y_test, y_pred, y_prob)

    # Extract feature names and coefficients
    try:
        clf_step = model.named_steps["clf"]
        prep_step = model.named_steps["preprocessor"]
        
        feature_names = []
        if numeric_cols:
            feature_names.extend(numeric_cols)
        if categorical_cols:
            cat_encoder = prep_step.named_transformers_["cat"].named_steps["encoder"]
            cat_names = list(cat_encoder.get_feature_names_out(categorical_cols))
            feature_names.extend(cat_names)
            
        coefs = clf_step.coef_[0]
        importance_df = pd.DataFrame({
            "Feature": feature_names[:len(coefs)],
            "Coefficient": coefs,
            "Abs_Coefficient": np.abs(coefs)
        }).sort_values(by="Abs_Coefficient", ascending=False).reset_index(drop=True)
    except Exception:
        importance_df = pd.DataFrame()

    return {
        "model": model,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "feature_importance": importance_df
    }


def train_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale"
) -> Dict[str, Any]:
    """
    Trains Support Vector Machine Classifier with the specified kernel.
    """
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = evaluate_classification(y_test, y_pred, y_prob)

    return {
        "model": model,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "kernel": kernel
    }


def compute_svm_decision_boundary_2d(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: List[str],
    kernel: str = "rbf",
    C: float = 1.0,
    max_samples: int = 1000
) -> Dict[str, Any]:
    """
    Projects numeric features to 2 principal components via PCA and fits SVM
    to generate a 2D decision boundary mesh for visualization.
    """
    # Sample if large
    if len(X_train) > max_samples:
        indices = np.random.RandomState(42).choice(len(X_train), max_samples, replace=False)
        X_sub = X_train.iloc[indices][numeric_cols].copy()
        y_sub = y_train.iloc[indices].copy()
    else:
        X_sub = X_train[numeric_cols].copy()
        y_sub = y_train.copy()

    # Impute and Scale
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_clean = scaler.fit_transform(imputer.fit_transform(X_sub))

    # 2D PCA
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_clean)

    # Fit SVM on 2D
    clf_2d = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    clf_2d.fit(X_2d, y_sub)

    # Create meshgrid
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    Z = clf_2d.predict(grid_points)
    Z = Z.reshape(xx.shape)

    explained_var = pca.explained_variance_ratio_

    return {
        "xx": xx,
        "yy": yy,
        "Z": Z,
        "X_2d": X_2d,
        "y": y_sub.values,
        "explained_var": explained_var,
        "kernel": kernel
    }


def train_knn_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    k: int = 5,
    metric: str = "minkowski"
) -> Dict[str, Any]:
    """
    Trains KNN Classifier with safety check on k <= len(X_train).
    """
    safe_k = max(1, min(k, len(X_train)))
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)
    
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", KNeighborsClassifier(n_neighbors=safe_k, metric=metric))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = evaluate_classification(y_test, y_pred, y_prob)

    return {
        "model": model,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "k": safe_k
    }


def run_knn_k_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    k_range: range = range(1, 21),
    max_train_samples: int = 5000
) -> pd.DataFrame:
    """
    Compares KNN Classifier across K values (e.g. 1 to 20) and records Accuracy, F1, and Precision.
    Uses subsample if training set is huge for fast interactive responsiveness.
    """
    if len(X_train) > max_train_samples:
        indices = np.random.RandomState(42).choice(len(X_train), max_train_samples, replace=False)
        X_tr = X_train.iloc[indices]
        y_tr = y_train.iloc[indices]
    else:
        X_tr = X_train
        y_tr = y_train

    # Pre-transform features once to avoid redundant preprocessing in loop
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)
    X_tr_trans = preprocessor.fit_transform(X_tr)
    X_te_trans = preprocessor.transform(X_test)

    records = []
    max_k = min(len(X_tr), max(k_range))
    actual_k_range = [k for k in k_range if 1 <= k <= max_k]

    for k in actual_k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_tr_trans, y_tr)
        y_pred = knn.predict(X_te_trans)
        
        m = evaluate_classification(y_test, y_pred)
        records.append({
            "K": k,
            "Accuracy": m["accuracy"],
            "F1_Score": m["f1_score"],
            "Precision": m["precision"],
            "Recall": m["recall"]
        })

    return pd.DataFrame(records)


def train_adaboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    n_estimators: int = 50,
    learning_rate: float = 1.0
) -> Dict[str, Any]:
    """
    Trains AdaBoost Classifier with Decision Stump weak learners.
    """
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)
    stump = DecisionTreeClassifier(max_depth=1, random_state=42)
    
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", AdaBoostClassifier(
            estimator=stump,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = evaluate_classification(y_test, y_pred, y_prob)

    # Feature importances
    try:
        clf_step = model.named_steps["clf"]
        prep_step = model.named_steps["preprocessor"]
        
        feature_names = []
        if numeric_cols:
            feature_names.extend(numeric_cols)
        if categorical_cols:
            cat_encoder = prep_step.named_transformers_["cat"].named_steps["encoder"]
            cat_names = list(cat_encoder.get_feature_names_out(categorical_cols))
            feature_names.extend(cat_names)
            
        importances = clf_step.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": feature_names[:len(importances)],
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    except Exception:
        importance_df = pd.DataFrame()

    return {
        "model": model,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "feature_importance": importance_df
    }
