"""
Expectation-Maximization (EM) Imputation Module using Gaussian Mixture Models (GMM).
Demonstrates mathematically defensible model-based missing value estimation
and compares against simple baseline median imputation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_absolute_error, mean_squared_error


class EMGMMImputer:
    """
    Expectation-Maximization Imputer using Gaussian Mixture Models (GMM).
    
    Mathematical Formulation:
    - E-step: Computes component responsibilities gamma_ik = P(Z_i = k | x_i, theta).
    - M-step: Re-estimates mixture weights pi_k, component means mu_k, and covariances Sigma_k.
    - Imputation: Uses conditional expectation E[x_miss | x_obs] = sum_k gamma_ik * (mu_k_miss + Sigma_k_mo * inv(Sigma_k_oo) * (x_obs - mu_k_obs)).
    """
    def __init__(self, n_components: int = 3, max_iter: int = 20, tol: float = 1e-3, random_state: int = 42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.gmm = None
        self.history_ = []

    def fit_transform(self, X_incomplete: np.ndarray) -> np.ndarray:
        """
        Fits GMM iteratively and imputes missing (NaN) values using conditional expectations.
        """
        X = X_incomplete.copy()
        nan_mask = np.isnan(X)
        if not np.any(nan_mask):
            return X

        n_samples, n_features = X.shape

        # Initial warm-start: fill NaNs with column medians
        col_medians = np.nanmedian(X, axis=0)
        # Handle all-nan columns edge case
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        inds = np.where(nan_mask)
        X_imputed = X.copy()
        X_imputed[inds] = np.take(col_medians, inds[1])

        prev_imputed_vals = X_imputed[nan_mask].copy()

        for iteration in range(self.max_iter):
            # 1. Fit GMM on current imputed matrix (E-step & M-step of GMM)
            self.gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type="full",
                max_iter=50,
                random_state=self.random_state + iteration,
                reg_covar=1e-4
            )
            self.gmm.fit(X_imputed)

            # Responsibilities for all samples: gamma_ik = P(k | x_i)
            responsibilities = self.gmm.predict_proba(X_imputed)

            # 2. Impute missing entries sample-by-sample via conditional Gaussian expectations
            for i in range(n_samples):
                row_mask = nan_mask[i]
                if not np.any(row_mask):
                    continue

                miss_idx = np.where(row_mask)[0]
                obs_idx = np.where(~row_mask)[0]

                if len(obs_idx) == 0:
                    # All features missing for this row: use global mixture mean
                    X_imputed[i, miss_idx] = self.gmm.means_[:, miss_idx].T @ responsibilities[i]
                    continue

                x_obs = X_imputed[i, obs_idx]
                imputed_row_miss = np.zeros(len(miss_idx))

                for k in range(self.n_components):
                    mu_k = self.gmm.means_[k]
                    sigma_k = self.gmm.covariances_[k]

                    mu_obs = mu_k[obs_idx]
                    mu_miss = mu_k[miss_idx]

                    sigma_oo = sigma_k[np.ix_(obs_idx, obs_idx)]
                    sigma_mo = sigma_k[np.ix_(miss_idx, obs_idx)]

                    # Regularized inverse for numerical stability
                    inv_sigma_oo = np.linalg.pinv(sigma_oo)

                    # Conditional expectation E[x_miss | x_obs, k]
                    cond_mean_k = mu_miss + sigma_mo @ inv_sigma_oo @ (x_obs - mu_obs)
                    imputed_row_miss += responsibilities[i, k] * cond_mean_k

                X_imputed[i, miss_idx] = imputed_row_miss

            # Check convergence
            curr_imputed_vals = X_imputed[nan_mask]
            delta = np.max(np.abs(curr_imputed_vals - prev_imputed_vals))
            self.history_.append(delta)

            if delta < self.tol:
                break
            prev_imputed_vals = curr_imputed_vals.copy()

        return X_imputed


def run_em_imputation_experiment(
    df: pd.DataFrame,
    target_feature: str = "early_avg_score",
    context_features: Optional[List[str]] = None,
    missing_rate: float = 0.20,
    n_components: int = 3,
    sample_size: int = 4000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Conducts a controlled laboratory experiment comparing EM-GMM Imputation vs Median Imputation.
    
    1. Selects complete student records where target_feature is observed.
    2. Introduces controlled synthetic missingness (e.g. 20% MCAR).
    3. Runs Median Imputation baseline.
    4. Runs EM Gaussian Mixture Model Imputation.
    5. Computes ground-truth reconstruction error (MAE, MSE, RMSE).
    6. Compares distribution statistics (Mean, Std, Skewness, Min, Max).
    """
    if context_features is None:
        context_features = [
            "early_vle_clicks",
            "early_active_days",
            "early_unique_sites",
            "studied_credits"
        ]

    features_all = [target_feature] + [f for f in context_features if f != target_feature]

    # Subset complete data
    complete_df = df[features_all].dropna().copy()
    if len(complete_df) > sample_size:
        complete_df = complete_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    X_true = complete_df.values.copy()
    n_samples, n_cols = X_true.shape
    target_col_idx = features_all.index(target_feature)

    # Introduce synthetic missingness
    rng = np.random.RandomState(random_state)
    missing_mask = rng.rand(n_samples) < missing_rate
    # Ensure at least some are missing
    if not np.any(missing_mask):
        missing_mask[rng.choice(n_samples, int(missing_rate * n_samples), replace=False)] = True

    X_masked = X_true.copy()
    X_masked[missing_mask, target_col_idx] = np.nan

    true_values = X_true[missing_mask, target_col_idx]

    # 1. Baseline: Median Imputation
    median_val = np.nanmedian(X_masked[:, target_col_idx])
    X_median = X_masked.copy()
    X_median[missing_mask, target_col_idx] = median_val
    median_imputed_values = X_median[missing_mask, target_col_idx]

    # 2. EM-GMM Imputation
    em_imputer = EMGMMImputer(n_components=n_components, max_iter=20, random_state=random_state)
    X_em = em_imputer.fit_transform(X_masked)
    em_imputed_values = X_em[missing_mask, target_col_idx]

    # Metrics computation on held-out masked entries
    def calc_metrics(y_t, y_p):
        mae = float(mean_absolute_error(y_t, y_p))
        mse = float(mean_squared_error(y_t, y_p))
        rmse = float(np.sqrt(mse))
        return {"MAE": round(mae, 3), "MSE": round(mse, 3), "RMSE": round(rmse, 3)}

    median_metrics = calc_metrics(true_values, median_imputed_values)
    em_metrics = calc_metrics(true_values, em_imputed_values)

    # Distribution comparison
    stats = []
    for name, vals in [
        ("Ground Truth", true_values),
        ("Median Imputation", median_imputed_values),
        ("EM-GMM Imputation", em_imputed_values)
    ]:
        stats.append({
            "Method": name,
            "Mean": round(float(np.mean(vals)), 2),
            "Std Dev": round(float(np.std(vals)), 2),
            "Min": round(float(np.min(vals)), 2),
            "Max": round(float(np.max(vals)), 2),
            "Median": round(float(np.median(vals)), 2)
        })

    stats_df = pd.DataFrame(stats)

    # Sample comparison rows
    comparison_sample = pd.DataFrame({
        "Ground_Truth": true_values[:15].round(2),
        "Median_Estimate": median_imputed_values[:15].round(2),
        "EM_GMM_Estimate": em_imputed_values[:15].round(2),
        "Median_Error": np.abs(true_values[:15] - median_imputed_values[:15]).round(2),
        "EM_GMM_Error": np.abs(true_values[:15] - em_imputed_values[:15]).round(2)
    })

    return {
        "target_feature": target_feature,
        "features": features_all,
        "missing_count": int(np.sum(missing_mask)),
        "total_count": n_samples,
        "missing_rate": missing_rate,
        "median_metrics": median_metrics,
        "em_metrics": em_metrics,
        "stats_df": stats_df,
        "comparison_sample": comparison_sample,
        "true_values": true_values,
        "median_values": median_imputed_values,
        "em_values": em_imputed_values,
        "convergence_history": em_imputer.history_
    }
