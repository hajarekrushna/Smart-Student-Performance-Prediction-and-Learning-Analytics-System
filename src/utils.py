"""
Visualization and UI Utility Helpers for Streamlit Dashboard.
Provides dark-themed, publication-quality Matplotlib and Plotly charts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

# Set dark theme palette for matplotlib
DARK_BG = "#0b1120"
CARD_BG = "#131d36"
TEXT_COLOR = "#f1f5f9"
GRID_COLOR = "#233358"
ACCENT_BLUE = "#3b82f6"
ACCENT_CYAN = "#06b6d4"
ACCENT_GREEN = "#10b981"
ACCENT_AMBER = "#f59e0b"
ACCENT_RED = "#ef4444"
ACCENT_PURPLE = "#a855f7"

PALETTE = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED, ACCENT_PURPLE, ACCENT_CYAN]


def apply_dark_theme(fig, ax=None):
    """Applies clean dark theme styling to matplotlib figures."""
    fig.patch.set_facecolor(CARD_BG)
    if ax is not None:
        if isinstance(ax, np.ndarray):
            for a in ax.ravel():
                _style_single_ax(a)
        else:
            _style_single_ax(ax)


def _style_single_ax(a):
    a.set_facecolor(CARD_BG)
    a.tick_params(colors=TEXT_COLOR, labelsize=9)
    a.xaxis.label.set_color(TEXT_COLOR)
    a.yaxis.label.set_color(TEXT_COLOR)
    a.title.set_color(TEXT_COLOR)
    for spine in a.spines.values():
        spine.set_color(GRID_COLOR)
    a.grid(True, linestyle="--", alpha=0.3, color=GRID_COLOR)


def plot_confusion_matrix(cm: np.ndarray, labels: List[str] = ["Not At-Risk", "At-Risk"]):
    """Plots dark-themed annotated confusion matrix."""
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    apply_dark_theme(fig, ax)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold", "color": "#ffffff"}
    )
    ax.set_ylabel("Actual Label", fontsize=10, weight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10, weight="bold")
    ax.set_title("Confusion Matrix", fontsize=12, weight="bold", pad=10)
    plt.tight_layout()
    return fig


def plot_roc_curve(fpr, tpr, roc_auc: float, model_name: str = "Model"):
    """Plots ROC curve with diagonal reference."""
    fig, ax = plt.subplots(figsize=(5, 3.8))
    apply_dark_theme(fig, ax)

    ax.plot(fpr, tpr, color=ACCENT_CYAN, lw=2.5, label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#64748b", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=10)
    ax.set_title(f"ROC Curve: {model_name}", fontsize=11, weight="bold", pad=10)
    ax.legend(loc="lower right", facecolor=DARK_BG, edgecolor=GRID_COLOR, fontsize=8, labelcolor=TEXT_COLOR)
    plt.tight_layout()
    return fig


def plot_svm_decision_boundary(data_2d: Dict[str, Any]):
    """Plots 2D PCA decision boundary contour and scatter points for SVM."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    apply_dark_theme(fig, ax)

    xx = data_2d["xx"]
    yy = data_2d["yy"]
    Z = data_2d["Z"]
    X_2d = data_2d["X_2d"]
    y = data_2d["y"]
    kernel = data_2d.get("kernel", "RBF")
    ev = data_2d.get("explained_var", [0.4, 0.3])

    # Contour filled
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=["#1e3a8a", "#831843"], alpha=0.35)
    ax.contour(xx, yy, Z, levels=[0.5], colors=["#38bdf8"], linewidths=2.0, linestyles="--")

    # Scatter points
    scatter0 = ax.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], c="#38bdf8", s=25, alpha=0.6, label="Not At-Risk (Pass/Dist)")
    scatter1 = ax.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], c="#f43f5e", s=25, alpha=0.6, label="At-Risk (Fail/Withdrawn)")

    ax.set_xlabel(f"Principal Component 1 ({ev[0]*100:.1f}% var)", fontsize=10)
    ax.set_ylabel(f"Principal Component 2 ({ev[1]*100:.1f}% var)", fontsize=10)
    ax.set_title(f"SVM Decision Boundary ({kernel.upper()} Kernel in 2D PCA Space)", fontsize=11, weight="bold")
    ax.legend(loc="upper right", facecolor=DARK_BG, edgecolor=GRID_COLOR, fontsize=8, labelcolor=TEXT_COLOR)
    plt.tight_layout()
    return fig


def plot_regression_actual_vs_predicted(y_true, y_pred, model_name: str = "Linear Regression"):
    """Plots actual vs predicted scatter plot with 45-degree reference line."""
    fig, ax = plt.subplots(figsize=(5.2, 4))
    apply_dark_theme(fig, ax)

    ax.scatter(y_true, y_pred, color=ACCENT_BLUE, alpha=0.45, s=20, edgecolors="none")

    # Reference diagonal
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], color=ACCENT_AMBER, lw=2, linestyle="--", label="Ideal Fit (y = x)")

    ax.set_xlabel("Actual Assessment Score", fontsize=10)
    ax.set_ylabel("Predicted Assessment Score", fontsize=10)
    ax.set_title(f"Actual vs Predicted: {model_name}", fontsize=11, weight="bold")
    ax.legend(loc="upper left", facecolor=DARK_BG, edgecolor=GRID_COLOR, fontsize=8, labelcolor=TEXT_COLOR)
    plt.tight_layout()
    return fig


def plot_residuals(residuals, y_pred, model_name: str = "Linear Regression"):
    """Plots residuals vs predicted score and residual distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    apply_dark_theme(fig, np.array([ax1, ax2]))

    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, color=ACCENT_PURPLE, alpha=0.4, s=18)
    ax1.axhline(0, color=ACCENT_AMBER, linestyle="--", lw=1.5)
    ax1.set_xlabel("Predicted Score", fontsize=10)
    ax1.set_ylabel("Residual (Actual - Predicted)", fontsize=10)
    ax1.set_title("Residuals vs Predicted", fontsize=11, weight="bold")

    # Residual Histogram
    ax2.hist(residuals, bins=25, color=ACCENT_CYAN, edgecolor=DARK_BG, alpha=0.7)
    ax2.axvline(0, color=ACCENT_AMBER, linestyle="--", lw=1.5)
    ax2.set_xlabel("Residual Value", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.set_title("Residual Distribution (Normality Check)", fontsize=11, weight="bold")

    plt.tight_layout()
    return fig


def plot_knn_curve(df_curve: pd.DataFrame, is_classification: bool = True):
    """Plots accuracy/F1 or RMSE/MAE vs K for KNN."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    apply_dark_theme(fig, ax)

    if is_classification:
        ax.plot(df_curve["K"], df_curve["Accuracy"], marker="o", color=ACCENT_CYAN, lw=2, label="Accuracy")
        ax.plot(df_curve["K"], df_curve["F1_Score"], marker="s", color=ACCENT_GREEN, lw=2, label="F1-Score")
        ax.set_ylabel("Score", fontsize=10)
        ax.set_title("KNN Classifier Performance vs K", fontsize=11, weight="bold")
    else:
        ax.plot(df_curve["K"], df_curve["RMSE"], marker="o", color=ACCENT_AMBER, lw=2, label="RMSE")
        ax.plot(df_curve["K"], df_curve["MAE"], marker="s", color=ACCENT_CYAN, lw=2, label="MAE")
        ax.set_ylabel("Error (Lower is Better)", fontsize=10)
        ax.set_title("KNN Regressor Error vs K", fontsize=11, weight="bold")

    ax.set_xlabel("K (Number of Neighbors)", fontsize=10)
    ax.set_xticks(df_curve["K"].values)
    ax.legend(loc="best", facecolor=DARK_BG, edgecolor=GRID_COLOR, fontsize=9, labelcolor=TEXT_COLOR)
    plt.tight_layout()
    return fig


def plot_clustering_pca(clust_res: Dict[str, Any], method: str = "kmeans"):
    """Plots 2D PCA projection of student clusters with centroids/medoids."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    apply_dark_theme(fig, ax)

    X_2d = clust_res["pca_data"]["X_2d"]
    ev = clust_res["pca_data"]["explained_variance"]
    data = clust_res[method]
    labels = data["labels"]
    centers_2d = data["centers_2d"]
    n_clusters = len(np.unique(labels))

    colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED, ACCENT_PURPLE, ACCENT_CYAN]

    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=colors[k % len(colors)],
            alpha=0.5,
            s=22,
            label=f"Cluster {k} (n={np.sum(mask)})"
        )

    # Plot Centers / Medoids
    center_label = "Centroids" if method == "kmeans" else "Medoids"
    ax.scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        c="#ffffff",
        marker="X",
        s=140,
        edgecolors="#000000",
        lw=1.5,
        label=f"Cluster {center_label}"
    )

    title_method = "K-Means" if method == "kmeans" else "K-Medoids (PAM)"
    ax.set_xlabel(f"PCA Dimension 1 ({ev[0]*100:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PCA Dimension 2 ({ev[1]*100:.1f}% variance)", fontsize=10)
    ax.set_title(f"{title_method} Clusters in 2D PCA Space", fontsize=11, weight="bold")
    ax.legend(loc="upper right", facecolor=DARK_BG, edgecolor=GRID_COLOR, fontsize=8, labelcolor=TEXT_COLOR)
    plt.tight_layout()
    return fig


def plot_hmm_engagement_timeline(seq_df: pd.DataFrame, decoded_states: np.ndarray):
    """
    Plots dual-panel HMM visualization:
    Top: Weekly VLE Clicks bar chart
    Bottom: Decoded Latent Engagement State timeline with state bands
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    apply_dark_theme(fig, np.array([ax1, ax2]))

    weeks = seq_df["week"].values
    clicks = seq_df["sum_click"].values

    state_colors = ["#ef4444", "#f59e0b", "#3b82f6", "#10b981"]
    state_names = ["Inactive", "Low", "Moderate", "High"]

    # 1. Weekly Clicks Bar Chart
    bar_colors = [state_colors[s] for s in decoded_states]
    ax1.bar(weeks, clicks, color=bar_colors, width=0.75, edgecolor=DARK_BG, alpha=0.85)
    ax1.set_ylabel("Weekly Clicks", fontsize=10, weight="bold")
    ax1.set_title("Observed Weekly VLE Interaction Activity", fontsize=11, weight="bold")

    # 2. Decoded Latent State Sequence
    ax2.step(weeks, decoded_states, where="mid", color="#38bdf8", lw=2.5, marker="o", markersize=4)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(state_names, fontsize=9, weight="bold")
    ax2.set_ylabel("Engagement State", fontsize=10, weight="bold")
    ax2.set_xlabel("Course Week (Relative to Start)", fontsize=10, weight="bold")
    ax2.set_title("Decoded Latent Engagement State (Viterbi Algorithm)", fontsize=11, weight="bold")

    # Add background state bands
    ax2.axhspan(-0.5, 0.5, color="#ef4444", alpha=0.15)
    ax2.axhspan(0.5, 1.5, color="#f59e0b", alpha=0.15)
    ax2.axhspan(1.5, 2.5, color="#3b82f6", alpha=0.15)
    ax2.axhspan(2.5, 3.5, color="#10b981", alpha=0.15)
    ax2.set_ylim([-0.5, 3.5])

    plt.tight_layout()
    return fig


def plot_em_imputation_comparison(true_vals, median_vals, em_vals, feature_name: str = "Feature"):
    """
    Plots distribution comparison: Ground Truth vs Median Imputation vs EM-GMM Imputation.
    Shows how median imputation distorts distribution variance and EM-GMM preserves it.
    """
    fig, ax = plt.subplots(figsize=(7, 3.8))
    apply_dark_theme(fig, ax)

    sns.kdeplot(true_vals, ax=ax, color="#ffffff", lw=2.2, label="Ground Truth (Actual)", linestyle="-")
    sns.kdeplot(em_vals, ax=ax, color=ACCENT_CYAN, lw=2.0, label="EM-GMM Imputation", linestyle="--")
    # For median, plot histogram or thin line since variance collapses
    ax.axvline(np.median(true_vals), color=ACCENT_AMBER, lw=2.5, linestyle=":", label="Median Imputation (Point Mass)")

    ax.set_xlabel(f"{feature_name} Value", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.set_title(f"Imputation Distribution Preservation: {feature_name}", fontsize=11, weight="bold")
    ax.legend(loc="upper right", facecolor=DARK_BG, edgecolor=GRID_COLOR, fontsize=8, labelcolor=TEXT_COLOR)
    plt.tight_layout()
    return fig
