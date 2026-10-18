"""
Clustering Models Module for Experiment 6.
Implements:
- K-Means Clustering
- K-Medoids Clustering using PAM (Partitioning Around Medoids)
- 2D PCA Visualization
- Cluster profiling and Persona Interpretation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist


class PAMKMedoids:
    """
    Partitioning Around Medoids (PAM) implementation in pure NumPy / SciPy.
    Guarantees cross-platform execution on Windows without external C-compiler dependencies.
    """
    def __init__(self, n_clusters: int = 3, max_iter: int = 100, metric: str = "euclidean", random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.metric = metric
        self.random_state = random_state
        self.medoid_indices_ = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X: np.ndarray):
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        k = min(self.n_clusters, n_samples)

        # 1. Initialize medoids using k-means++ heuristic
        first_medoid = rng.choice(n_samples)
        medoids = [first_medoid]
        dist_matrix = cdist(X, X, metric=self.metric)

        for _ in range(1, k):
            current_dists = dist_matrix[:, medoids].min(axis=1)
            probs = current_dists / (current_dists.sum() + 1e-12)
            next_medoid = rng.choice(n_samples, p=probs)
            medoids.append(next_medoid)

        medoids = np.array(medoids)

        # 2. Iterate PAM swap phase
        labels = np.zeros(n_samples, dtype=int)
        for _ in range(self.max_iter):
            # Assignment phase
            dists_to_medoids = dist_matrix[:, medoids]
            labels = np.argmin(dists_to_medoids, axis=1)

            # Update phase: find point in each cluster minimizing total distance
            new_medoids = medoids.copy()
            for c in range(k):
                cluster_points = np.where(labels == c)[0]
                if len(cluster_points) == 0:
                    continue
                # Submatrix of distances among points in this cluster
                sub_dists = dist_matrix[np.ix_(cluster_points, cluster_points)]
                best_idx = cluster_points[np.argmin(sub_dists.sum(axis=1))]
                new_medoids[c] = best_idx

            if np.array_equal(new_medoids, medoids):
                break
            medoids = new_medoids

        self.medoid_indices_ = medoids
        self.cluster_centers_ = X[medoids]
        self.labels_ = labels
        self.inertia_ = float(np.sum(dist_matrix[np.arange(n_samples), medoids[labels]]))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = cdist(X, self.cluster_centers_, metric=self.metric)
        return np.argmin(dists, axis=1)


def prepare_clustering_data(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    sample_size: Optional[int] = 3000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler, List[str]]:
    """
    Extracts features for clustering, imputes, scales, and optionally subsamples
    for manageable K-Medoids pairwise distance computation.
    """
    if features is None:
        features = [
            "early_vle_clicks",
            "early_active_days",
            "early_unique_sites",
            "early_clicks_per_active_day",
            "early_assessment_count",
            "early_avg_score",
            "studied_credits",
            "num_of_prev_attempts"
        ]

    df_subset = df[features].copy()

    # Subsample if requested
    if sample_size and len(df_subset) > sample_size:
        df_sampled = df_subset.sample(n=sample_size, random_state=random_state).copy()
    else:
        df_sampled = df_subset.copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(df_sampled)
    X_scaled = scaler.fit_transform(X_imputed)

    return df_sampled, X_scaled, scaler, features


def run_clustering_models(
    df: pd.DataFrame,
    n_clusters: int = 3,
    sample_size: int = 2500,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Runs both K-Means and PAM K-Medoids on the student engagement dataset.
    Computes silhouette scores, Davies-Bouldin index, 2D PCA projections, and cluster profiles.
    """
    df_raw, X_scaled, scaler, feature_names = prepare_clustering_data(
        df, sample_size=sample_size, random_state=random_state
    )

    # 1. Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X_scaled)
    km_labels = kmeans.labels_

    # 2. Fit PAM K-Medoids
    kmedoids = PAMKMedoids(n_clusters=n_clusters, max_iter=50, random_state=random_state)
    kmedoids.fit(X_scaled)
    kmed_labels = kmedoids.labels_

    # Compute Validation Metrics
    km_sil = float(silhouette_score(X_scaled, km_labels)) if len(np.unique(km_labels)) > 1 else 0.0
    kmed_sil = float(silhouette_score(X_scaled, kmed_labels)) if len(np.unique(kmed_labels)) > 1 else 0.0

    km_db = float(davies_bouldin_score(X_scaled, km_labels)) if len(np.unique(km_labels)) > 1 else 0.0
    kmed_db = float(davies_bouldin_score(X_scaled, kmed_labels)) if len(np.unique(kmed_labels)) > 1 else 0.0

    # 2D PCA Projection
    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X_scaled)
    km_centers_2d = pca.transform(kmeans.cluster_centers_)
    kmed_centers_2d = pca.transform(kmedoids.cluster_centers_)

    # Cluster Profiles (unscaled mean values per cluster)
    df_km = df_raw.copy()
    df_km["Cluster"] = km_labels
    km_profile = df_km.groupby("Cluster")[feature_names].mean().round(2)
    km_sizes = df_km["Cluster"].value_counts().sort_index()
    km_profile["Student_Count"] = km_sizes
    km_profile["Percent"] = (km_sizes / len(df_km) * 100).round(1)

    df_kmed = df_raw.copy()
    df_kmed["Cluster"] = kmed_labels
    kmed_profile = df_kmed.groupby("Cluster")[feature_names].mean().round(2)
    kmed_sizes = df_kmed["Cluster"].value_counts().sort_index()
    kmed_profile["Student_Count"] = kmed_sizes
    kmed_profile["Percent"] = (kmed_sizes / len(df_kmed) * 100).round(1)

    # Automated Persona Interpretation
    personas = interpret_clusters(km_profile, feature_names)

    return {
        "kmeans": {
            "model": kmeans,
            "labels": km_labels,
            "silhouette": km_sil,
            "davies_bouldin": km_db,
            "centers_2d": km_centers_2d,
            "profile": km_profile,
            "sizes": km_sizes
        },
        "kmedoids": {
            "model": kmedoids,
            "labels": kmed_labels,
            "silhouette": kmed_sil,
            "davies_bouldin": kmed_db,
            "centers_2d": kmed_centers_2d,
            "profile": kmed_profile,
            "sizes": kmed_sizes,
            "medoid_indices": kmedoids.medoid_indices_
        },
        "pca_data": {
            "X_2d": X_2d,
            "explained_variance": pca.explained_variance_ratio_
        },
        "personas": personas,
        "feature_names": feature_names,
        "sample_size": len(df_raw)
    }


def interpret_clusters(profile_df: pd.DataFrame, features: List[str]) -> Dict[int, str]:
    """
    Generates meaningful learning persona descriptions based on cluster profiles.
    """
    personas = {}
    clicks_col = "early_vle_clicks" if "early_vle_clicks" in profile_df.columns else features[0]
    score_col = "early_avg_score" if "early_avg_score" in profile_df.columns else features[1]

    mean_clicks = profile_df[clicks_col].mean()
    mean_score = profile_df[score_col].mean()

    for cluster_id, row in profile_df.iterrows():
        c = row[clicks_col]
        s = row[score_col]
        
        if c >= mean_clicks and s >= mean_score:
            persona = "High Achievers: High engagement, frequent VLE interaction, top assessment scores"
        elif c >= mean_clicks and s < mean_score:
            persona = "Diligent Strugglers: Active on VLE but struggling with assessment grades"
        elif c < mean_clicks and s >= mean_score:
            persona = "Efficient Learners: Low VLE activity but solid assessment performance"
        else:
            persona = "Disengaged Learners: Low VLE activity and low/missing assessment scores (high risk)"
            
        personas[int(cluster_id)] = persona

    return personas
