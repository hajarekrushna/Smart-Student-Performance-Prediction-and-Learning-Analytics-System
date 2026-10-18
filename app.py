"""
Smart Student Performance Prediction & Learning Analytics System
Interactive Streamlit Dashboard.
Implements Experiments 1 through 7 for Machine Learning Laboratory Mini Project.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add workspace directory to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import find_data_dir, inspect_all_tables, load_raw_csv
from src.preprocessing import (
    build_preprocessed_features,
    prepare_classification_dataset,
    prepare_regression_dataset
)
from src.classification_models import (
    train_logistic_regression,
    train_svm,
    train_knn_classifier,
    train_adaboost,
    compute_svm_decision_boundary_2d,
    run_knn_k_search
)
from src.regression_models import (
    train_linear_regression,
    train_polynomial_regression,
    train_knn_regressor,
    run_knn_regressor_k_search
)
from src.clustering_models import run_clustering_models
from src.hmm_model import (
    get_default_hmm_parameters,
    forward_algorithm,
    viterbi_algorithm,
    extract_student_sequence,
    STATE_LABELS,
    OBS_LABELS
)
from src.em_imputation import run_em_imputation_experiment
from src.evaluation import metrics_to_dataframe
from src.utils import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_svm_decision_boundary,
    plot_regression_actual_vs_predicted,
    plot_residuals,
    plot_knn_curve,
    plot_clustering_pca,
    plot_hmm_engagement_timeline,
    plot_em_imputation_comparison,
    DARK_BG,
    CARD_BG,
    ACCENT_BLUE,
    ACCENT_CYAN,
    ACCENT_GREEN,
    ACCENT_AMBER,
    ACCENT_RED
)

# Page Configuration
st.set_page_config(
    page_title="Smart Student Learning Analytics & Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sleek Dark Theme
st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #131d36 0%, #1e293b 100%);
        border: 1px solid #233358;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        margin-bottom: 15px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Glass Section Container */
    .glass-container {
        background-color: #131d36;
        border: 1px solid #233358;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 22px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    /* Badges */
    .badge-at-risk {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-safe {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #ffffff;
        font-weight: 700;
    }
    
    /* Dataframe styling */
    .dataframe {
        color: #f1f5f9 !important;
        background-color: #131d36 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_and_preprocess_dataset():
    """Loads and caches features and weekly engagement datasets."""
    data_dir = find_data_dir()
    df_feat, df_weekly = build_preprocessed_features(data_dir, cutoff_day=60)
    return data_dir, df_feat, df_weekly


# Load data with spinner
with st.spinner("Initializing Learning Analytics System and loading OULAD data..."):
    try:
        data_dir, df_feat, df_weekly = load_and_preprocess_dataset()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

# Sidebar Navigation
st.sidebar.markdown("""
<div style="text-align: center; padding: 12px 0 20px 0;">
    <h2 style="margin: 0; color: #38bdf8; font-size: 1.4rem;">🎓 Student Analytics</h2>
    <span style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em;">ML Lab Experiments Suite</span>
</div>
""", unsafe_allow_html=True)

nav_options = [
    "🌟 1. Overview & KPIs",
    "🔍 2. Dataset Explorer",
    "🎯 3. Classification (Exp 1, 2, 3, 7)",
    "📈 4. Regression (Exp 1, 3)",
    "🧩 5. Clustering: K-Means vs K-Medoids PAM (Exp 6)",
    "🔄 6. EM Missing Value Imputation (Exp 5)",
    "⏱️ 7. HMM Weekly Engagement (Exp 4)",
    "🏆 8. Comprehensive Model Comparison",
    "📖 9. About & Lab Mapping"
]

selected_nav = st.sidebar.radio("Navigation Menu", nav_options)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="font-size: 0.78rem; color: #64748b; line-height: 1.5;">
    <b>Cutoff Window:</b> Day 60 (Early Warning)<br>
    <b>Total Records:</b> {len(df_feat):,}<br>
    <b>Unique Students:</b> {df_feat['id_student'].nunique():,}<br>
    <b>Courses:</b> {df_feat['code_module'].nunique()} Modules ({df_feat['code_presentation'].nunique()} Presentations)
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# SECTION 1: OVERVIEW & KPIS
# ==============================================================================
if "1. Overview" in selected_nav:
    st.markdown("""
    <div style="margin-bottom: 25px;">
        <h1 style="margin: 0; font-size: 2.1rem; color: #ffffff;">Smart Student Performance Prediction & Learning Analytics</h1>
        <p style="color: #94a3b8; font-size: 1rem; margin-top: 6px;">
            Evidence-based early warning system and learning behavior analytics built on the Open University Learning Analytics Dataset (OULAD).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # High-level Metrics Row
    total_students = df_feat["id_student"].nunique()
    total_enrollments = len(df_feat)
    at_risk_count = int(df_feat["is_at_risk"].sum())
    at_risk_pct = (at_risk_count / total_enrollments) * 100
    safe_pct = 100 - at_risk_pct

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Enrollments</div>
            <div class="metric-value">{total_enrollments:,}</div>
            <div class="metric-sub">Across 7 courses & 4 cohorts</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Unique Students</div>
            <div class="metric-value">{total_students:,}</div>
            <div class="metric-sub">Anonymized learner profiles</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">At-Risk Rate</div>
            <div class="metric-value" style="color: #f87171;">{at_risk_pct:.1f}%</div>
            <div class="metric-sub">{at_risk_count:,} Fail or Withdrawn</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Early Warning Cutoff</div>
            <div class="metric-value" style="color: #38bdf8;">Day 60</div>
            <div class="metric-sub">Zero target leakage protocol</div>
        </div>
        """, unsafe_allow_html=True)

    # Detailed Overview Charts
    row2_col1, row2_col2 = st.columns([1.1, 1.2])

    with row2_col1:
        st.markdown("""
        <div class="glass-container">
            <h3 style="margin-top: 0; font-size: 1.15rem;">🎯 Final Outcome Distribution</h3>
            <p style="color: #94a3b8; font-size: 0.82rem;">Binary Risk Definition: At-Risk (Fail / Withdrawn) vs Not At-Risk (Pass / Distinction)</p>
        </div>
        """, unsafe_allow_html=True)
        
        fig_outcome, ax_outcome = plt.subplots(figsize=(5.2, 4.2))
        fig_outcome.patch.set_facecolor(CARD_BG)
        ax_outcome.set_facecolor(CARD_BG)
        
        outcome_counts = df_feat["final_result"].value_counts()
        colors = ["#10b981", "#ef4444", "#f59e0b", "#3b82f6"]
        wedges, texts, autotexts = ax_outcome.pie(
            outcome_counts.values,
            labels=outcome_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops={"edgecolor": DARK_BG, "linewidth": 2, "width": 0.65},
            textprops={"color": "#f1f5f9", "fontsize": 10, "weight": "bold"}
        )
        for at in autotexts:
            at.set_color("#ffffff")
        ax_outcome.set_title("OULAD Final Outcome Breakdown", color="#ffffff", fontsize=11, weight="bold")
        plt.tight_layout()
        st.pyplot(fig_outcome)

    with row2_col2:
        st.markdown("""
        <div class="glass-container">
            <h3 style="margin-top: 0; font-size: 1.15rem;">📊 Early Engagement by Risk Status (Day 0 - 60)</h3>
            <p style="color: #94a3b8; font-size: 0.82rem;">Comparing median VLE clicks and active study days before the day 60 cutoff</p>
        </div>
        """, unsafe_allow_html=True)

        fig_eng, (ax_e1, ax_e2) = plt.subplots(1, 2, figsize=(6.8, 3.8))
        for a in [ax_e1, ax_e2]:
            fig_eng.patch.set_facecolor(CARD_BG)
            a.set_facecolor(CARD_BG)
            a.tick_params(colors="#f1f5f9", labelsize=9)
            a.spines["bottom"].set_color("#233358")
            a.spines["left"].set_color("#233358")
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)

        # Clicks comparison
        clicks_by_risk = df_feat.groupby("is_at_risk")["early_vle_clicks"].median()
        ax_e1.bar(["Pass/Dist", "At-Risk"], clicks_by_risk.values, color=["#10b981", "#ef4444"], width=0.6)
        ax_e1.set_title("Median Early VLE Clicks", color="#ffffff", fontsize=10, weight="bold")
        ax_e1.set_ylabel("Clicks", color="#f1f5f9", fontsize=9)

        # Active days comparison
        days_by_risk = df_feat.groupby("is_at_risk")["early_active_days"].median()
        ax_e2.bar(["Pass/Dist", "At-Risk"], days_by_risk.values, color=["#10b981", "#ef4444"], width=0.6)
        ax_e2.set_title("Median Active Days", color="#ffffff", fontsize=10, weight="bold")
        ax_e2.set_ylabel("Days", color="#f1f5f9", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig_eng)

    # Early Warning Framework Explanation
    st.markdown("""
    <div class="glass-container">
        <h3 style="margin-top: 0; font-size: 1.15rem; color: #38bdf8;">🛡️ Academic Integrity: The Day 60 Observation Cutoff</h3>
        <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6;">
            A common failure mode in academic learning analytics is <b>target leakage</b> (e.g., using final exam scores or end-of-term activity to predict mid-term risk). 
            In this system, all predictive input features are strictly computed from interactions that occurred <b>on or before day 60</b> of the semester. 
            This replicates a realistic operational scenario where university counselors identify struggling students early enough to intervene.
        </p>
        <div style="display: flex; gap: 15px; margin-top: 15px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 200px; background: #0f172a; padding: 12px; border-radius: 8px; border-left: 4px solid #38bdf8;">
                <div style="font-weight: bold; color: #38bdf8; font-size: 0.85rem;">Input Features (&le; Day 60)</div>
                <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">Early VLE clicks, active days, early assessment submission delay, early grades, registration timing, credits.</div>
            </div>
            <div style="flex: 1; min-width: 200px; background: #0f172a; padding: 12px; border-radius: 8px; border-left: 4px solid #10b981;">
                <div style="font-weight: bold; color: #10b981; font-size: 0.85rem;">Classification Target</div>
                <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">At-Risk status: 1 if Fail or Withdrawn, 0 if Pass or Distinction.</div>
            </div>
            <div style="flex: 1; min-width: 200px; background: #0f172a; padding: 12px; border-radius: 8px; border-left: 4px solid #f59e0b;">
                <div style="font-weight: bold; color: #f59e0b; font-size: 0.85rem;">Regression Target (> Day 60)</div>
                <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">Average score on subsequent assessments submitted after day 60 (proxy for later academic performance).</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 2: DATASET EXPLORER
# ==============================================================================
elif "2. Dataset Explorer" in selected_nav:
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem;">🔍 OULAD Dataset Explorer</h1>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Inspect raw tables, schemas, missing values, and descriptive statistics. Anonymization is strictly preserved.
        </p>
    </div>
    """, unsafe_allow_html=True)

    table_summaries = inspect_all_tables(data_dir)
    table_names = list(table_summaries.keys())

    selected_table = st.selectbox("Select Table to Inspect", table_names, index=0)
    meta = table_summaries[selected_table]

    # Meta cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Table Name</div>
            <div class="metric-value" style="font-size: 1.3rem;">{selected_table}</div>
            <div class="metric-sub">Path: {meta['path']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Records</div>
            <div class="metric-value">{meta['total_rows']:,}</div>
            <div class="metric-sub">Rows in CSV</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">File Size</div>
            <div class="metric-value">{meta['size_mb']} MB</div>
            <div class="metric-sub">{len(meta['columns'])} Columns</div>
        </div>
        """, unsafe_allow_html=True)

    # Sample records & info
    st.markdown(f"### Sample Records: `{selected_table}`")
    st.dataframe(meta["sample_head"], use_container_width=True)

    # Column dtypes and null counts on a sample
    sample_df = load_raw_csv(data_dir, selected_table, nrows=5000)
    st.markdown("### Column Schema & Missingness Summary")
    schema_rows = []
    for col in sample_df.columns:
        null_count = sample_df[col].isnull().sum()
        null_pct = (null_count / len(sample_df)) * 100
        schema_rows.append({
            "Column": col,
            "Data Type": str(sample_df[col].dtype),
            "Null Count (Sample 5k)": null_count,
            "Null Percentage": f"{null_pct:.1f}%",
            "Unique Values": sample_df[col].nunique()
        })
    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)


# ==============================================================================
# SECTION 3: CLASSIFICATION (EXP 1, 2, 3, 7)
# ==============================================================================
elif "3. Classification" in selected_nav:
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem;">🎯 Experiment 1, 2, 3 & 7: Student Risk Classification</h1>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Predicting whether a learner is at risk of failing or withdrawing (<span class="badge-at-risk">At-Risk</span> vs <span class="badge-safe">Pass/Distinction</span>) 
            using solely early engagement metrics and demographics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Classification Controls
    c_ctrl1, c_ctrl2, c_ctrl3 = st.columns(3)
    with c_ctrl1:
        clf_algo = st.selectbox(
            "Select Machine Learning Algorithm",
            [
                "Logistic Regression (Exp 1)",
                "Support Vector Machine (Exp 2)",
                "K-Nearest Neighbors (Exp 3)",
                "AdaBoost Ensemble (Exp 7)"
            ]
        )
    with c_ctrl2:
        sample_size = st.slider("Training Sample Size", min_value=1000, max_value=25000, value=6000, step=1000,
                               help="Adjust sample size for fast interactive performance on student laptops.")
    with c_ctrl3:
        include_demo = st.checkbox("Include Demographics Features", value=True,
                                  help="Include gender, region, education, age, disability, and module code.")

    # Prepare dataset
    Xc, yc, num_cols, cat_cols = prepare_classification_dataset(df_feat, include_demographics=include_demo)
    
    # Subsample if requested
    if len(Xc) > sample_size:
        indices = np.random.RandomState(42).choice(len(Xc), sample_size, replace=False)
        Xc_active = Xc.iloc[indices].reset_index(drop=True)
        yc_active = yc.iloc[indices].reset_index(drop=True)
    else:
        Xc_active = Xc.copy()
        yc_active = yc.copy()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        Xc_active, yc_active, test_size=0.25, random_state=42, stratify=yc_active
    )

    # Algorithm specific hyperparameters
    if "Logistic Regression" in clf_algo:
        st.markdown("#### ⚙️ Logistic Regression Hyperparameters")
        h1, h2 = st.columns(2)
        with h1:
            C_val = st.slider("Regularization Strength C (Inverse)", 0.01, 10.0, 1.0, 0.05)
        with h2:
            max_iter = st.number_input("Max Iterations", 100, 2000, 1000, 100)

        with st.spinner("Training Logistic Regression Model..."):
            res = train_logistic_regression(X_train, y_train, X_test, y_test, num_cols, cat_cols, C=C_val, max_iter=max_iter)

    elif "Support Vector Machine" in clf_algo:
        st.markdown("#### ⚙️ SVM Hyperparameters")
        h1, h2, h3 = st.columns(3)
        with h1:
            kernel = st.selectbox("Kernel Function", ["rbf", "linear", "poly"], index=0)
        with h2:
            svm_c = st.slider("Penalty Parameter C", 0.1, 10.0, 1.0, 0.1)
        with h3:
            show_2d = st.checkbox("Show 2D Decision Boundary (PCA)", value=True)

        with st.spinner(f"Training Support Vector Machine ({kernel.upper()} kernel)..."):
            res = train_svm(X_train, y_train, X_test, y_test, num_cols, cat_cols, kernel=kernel, C=svm_c)

    elif "K-Nearest Neighbors" in clf_algo:
        st.markdown("#### ⚙️ KNN Classifier Hyperparameters")
        h1, h2, h3 = st.columns(3)
        with h1:
            knn_k = st.slider("K (Number of Neighbors)", 1, 25, 5, 1)
        with h2:
            metric = st.selectbox("Distance Metric", ["minkowski", "manhattan"], index=0)
        with h3:
            run_k_curve = st.checkbox("Run K-Search Analysis (K=1 to 20)", value=False)

        with st.spinner("Training KNN Classifier..."):
            res = train_knn_classifier(X_train, y_train, X_test, y_test, num_cols, cat_cols, k=knn_k, metric=metric)

    else:  # AdaBoost
        st.markdown("#### ⚙️ AdaBoost Ensemble Hyperparameters")
        h1, h2 = st.columns(2)
        with h1:
            n_est = st.slider("Number of Estimators (Decision Stumps)", 10, 150, 50, 10)
        with h2:
            lr = st.slider("Learning Rate", 0.1, 2.0, 1.0, 0.1)

        with st.spinner("Training AdaBoost Classifier..."):
            res = train_adaboost(X_train, y_train, X_test, y_test, num_cols, cat_cols, n_estimators=n_est, learning_rate=lr)

    # Display Metrics Cards
    m = res["metrics"]
    st.markdown("### 📊 Model Evaluation Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Accuracy</div>
            <div class="metric-value">{m['accuracy']*100:.1f}%</div>
            <div class="metric-sub">Overall correct</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Precision</div>
            <div class="metric-value">{m['precision']*100:.1f}%</div>
            <div class="metric-sub">At-risk precision</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Recall</div>
            <div class="metric-value">{m['recall']*100:.1f}%</div>
            <div class="metric-sub">At-risk captured</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">F1-Score</div>
            <div class="metric-value">{m['f1_score']:.3f}</div>
            <div class="metric-sub">Harmonic mean</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        roc_str = f"{m['roc_auc']:.3f}" if m['roc_auc'] is not None else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">ROC-AUC</div>
            <div class="metric-value">{roc_str}</div>
            <div class="metric-sub">Discriminative power</div>
        </div>
        """, unsafe_allow_html=True)

    # Diagnostic Plots Row
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown("#### Confusion Matrix")
        fig_cm = plot_confusion_matrix(m["confusion_matrix"])
        st.pyplot(fig_cm)

    with p_col2:
        st.markdown("#### ROC Curve")
        if m["roc_curve"] is not None:
            fpr, tpr = m["roc_curve"]
            fig_roc = plot_roc_curve(fpr, tpr, m["roc_auc"], model_name=clf_algo.split("(")[0])
            st.pyplot(fig_roc)
        else:
            st.info("ROC Curve not applicable.")

    # Additional Model-Specific Visualizations
    if "Support Vector Machine" in clf_algo and show_2d:
        st.markdown("#### 🌐 SVM Decision Boundary (2D PCA Space)")
        with st.spinner("Computing 2D PCA Decision Boundary Mesh..."):
            data_2d = compute_svm_decision_boundary_2d(X_train, y_train, num_cols, kernel=kernel, C=svm_c, max_samples=800)
            fig_bound = plot_svm_decision_boundary(data_2d)
            st.pyplot(fig_bound)

    if "K-Nearest Neighbors" in clf_algo and run_k_curve:
        st.markdown("#### 📈 KNN Performance vs K (K = 1 to 20)")
        with st.spinner("Evaluating KNN across K=1..20..."):
            df_k = run_knn_k_search(X_train, y_train, X_test, y_test, num_cols, cat_cols, k_range=range(1, 21), max_train_samples=3000)
            fig_k = plot_knn_curve(df_k, is_classification=True)
            st.pyplot(fig_k)
            st.dataframe(df_k, use_container_width=True)

    if "feature_importance" in res and not res["feature_importance"].empty:
        st.markdown("#### 🌟 Top Discriminative Features")
        st.dataframe(res["feature_importance"].head(10), use_container_width=True)

    # Download Predictions
    pred_export_df = pd.DataFrame({
        "Actual_Label": y_test.values,
        "Predicted_Label": res["y_pred"],
        "Risk_Probability": res["y_prob"][:, 1].round(4) if res["y_prob"] is not None else np.nan
    })
    csv_bytes = pred_export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Test Predictions (CSV)",
        data=csv_bytes,
        file_name=f"classification_predictions_{clf_algo.split()[0].lower()}.csv",
        mime="text/csv"
    )


# ==============================================================================
# SECTION 4: REGRESSION (EXP 1, 3)
# ==============================================================================
elif "4. Regression" in selected_nav:
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem;">📈 Experiment 1 & 3: Predicting Subsequent Assessment Scores</h1>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Predicting average score on later assessments (submitted <b>after day 60</b>) using early interaction features (&le; Day 60).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Regression Controls
    r_c1, r_c2 = st.columns(2)
    with r_c1:
        reg_algo = st.selectbox(
            "Select Regression Algorithm",
            [
                "Linear Regression (Exp 1)",
                "Polynomial Regression (Exp 1)",
                "KNN Regressor (Exp 3)"
            ]
        )
    with r_c2:
        sample_size_reg = st.slider("Sample Size", min_value=1000, max_value=21000, value=6000, step=1000)

    # Prepare Regression Data
    Xr, yr, num_cols_reg = prepare_regression_dataset(df_feat)
    if len(Xr) > sample_size_reg:
        indices = np.random.RandomState(42).choice(len(Xr), sample_size_reg, replace=False)
        Xr_active = Xr.iloc[indices].reset_index(drop=True)
        yr_active = yr.iloc[indices].reset_index(drop=True)
    else:
        Xr_active = Xr.copy()
        yr_active = yr.copy()

    from sklearn.model_selection import train_test_split
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        Xr_active, yr_active, test_size=0.25, random_state=42
    )

    if "Linear Regression" in reg_algo:
        with st.spinner("Fitting Linear Regression Model..."):
            res_reg = train_linear_regression(X_train_r, y_train_r, X_test_r, y_test_r, num_cols_reg)

    elif "Polynomial Regression" in reg_algo:
        deg = st.slider("Polynomial Degree", min_value=2, max_value=3, value=2,
                        help="Degree 2 creates quadratic and pairwise interaction terms.")
        with st.spinner(f"Fitting Polynomial Regression (Degree {deg})..."):
            res_reg = train_polynomial_regression(X_train_r, y_train_r, X_test_r, y_test_r, num_cols_reg, degree=deg)
            st.info(f"Generated {res_reg['poly_features_count']} polynomial interaction features.")

    else:  # KNN Regressor
        k_val = st.slider("K (Number of Neighbors)", 1, 25, 7, 1)
        run_k_reg_curve = st.checkbox("Run K-Search Error Curve (K=1 to 20)", value=False)
        with st.spinner("Fitting KNN Regressor..."):
            res_reg = train_knn_regressor(X_train_r, y_train_r, X_test_r, y_test_r, num_cols_reg, k=k_val)

    # Display Metrics Cards
    mr = res_reg["metrics"]
    st.markdown("### 📊 Regression Performance Metrics")
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MAE (Mean Absolute Error)</div>
            <div class="metric-value">{mr['mae']:.2f}</div>
            <div class="metric-sub">Average score deviation</div>
        </div>
        """, unsafe_allow_html=True)
    with rc2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">RMSE (Root Mean Sq Error)</div>
            <div class="metric-value">{mr['rmse']:.2f}</div>
            <div class="metric-sub">Penalizes large errors</div>
        </div>
        """, unsafe_allow_html=True)
    with rc3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MSE (Mean Squared Error)</div>
            <div class="metric-value">{mr['mse']:.2f}</div>
            <div class="metric-sub">Variance of residuals</div>
        </div>
        """, unsafe_allow_html=True)
    with rc4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">R² Score</div>
            <div class="metric-value">{mr['r2']:.3f}</div>
            <div class="metric-sub">Explained variance ratio</div>
        </div>
        """, unsafe_allow_html=True)

    # Visualizations
    p_reg1, p_reg2 = st.columns(2)
    with p_reg1:
        st.markdown("#### Actual vs Predicted Scores")
        fig_avp = plot_regression_actual_vs_predicted(y_test_r.values, res_reg["y_pred"], model_name=reg_algo.split("(")[0])
        st.pyplot(fig_avp)

    with p_reg2:
        st.markdown("#### Residual Diagnostics")
        fig_res = plot_residuals(res_reg["residuals"], res_reg["y_pred"], model_name=reg_algo.split("(")[0])
        st.pyplot(fig_res)

    if "KNN Regressor" in reg_algo and run_k_reg_curve:
        st.markdown("#### 📉 KNN Error Curve vs K (1 to 20)")
        with st.spinner("Computing KNN Regression error curve..."):
            df_k_reg = run_knn_regressor_k_search(X_train_r, y_train_r, X_test_r, y_test_r, num_cols_reg, k_range=range(1, 21))
            fig_k_r = plot_knn_curve(df_k_reg, is_classification=False)
            st.pyplot(fig_k_r)
            st.dataframe(df_k_reg, use_container_width=True)

    if "coefficients" in res_reg:
        st.markdown("#### 🔍 Linear Regression Feature Coefficients")
        st.dataframe(res_reg["coefficients"], use_container_width=True)

    # Download Predictions
    reg_export_df = pd.DataFrame({
        "Actual_Score": y_test_r.values.round(2),
        "Predicted_Score": res_reg["y_pred"].round(2),
        "Absolute_Error": np.abs(y_test_r.values - res_reg["y_pred"]).round(2)
    })
    csv_bytes_r = reg_export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Regression Predictions (CSV)",
        data=csv_bytes_r,
        file_name=f"regression_predictions_{reg_algo.split()[0].lower()}.csv",
        mime="text/csv"
    )


# ==============================================================================
# SECTION 5: CLUSTERING: K-MEANS VS K-MEDOIDS PAM (EXP 6)
# ==============================================================================
elif "5. Clustering" in selected_nav:
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem;">🧩 Experiment 6: Learner Segmentation (K-Means vs K-Medoids PAM)</h1>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Unsupervised discovery of student engagement personas using behavioral and academic metrics (strictly without outcome labels).
        </p>
    </div>
    """, unsafe_allow_html=True)

    cl_c1, cl_c2 = st.columns(2)
    with cl_c1:
        n_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=6, value=3)
    with cl_c2:
        cl_sample = st.slider("Clustering Sample Size", min_value=500, max_value=5000, value=2000,
                              help="Manageable sample ensures fast pairwise distance calculations for PAM K-Medoids.")

    with st.spinner("Running K-Means and PAM K-Medoids Clustering..."):
        clust_res = run_clustering_models(df_feat, n_clusters=n_clusters, sample_size=cl_sample)

    # Silhouette and Davies Bouldin Cards
    km_data = clust_res["kmeans"]
    kmed_data = clust_res["kmedoids"]

    m_cl1, m_cl2, m_cl3, m_cl4 = st.columns(4)
    with m_cl1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">K-Means Silhouette</div>
            <div class="metric-value">{km_data['silhouette']:.3f}</div>
            <div class="metric-sub">Separation quality (-1 to 1)</div>
        </div>
        """, unsafe_allow_html=True)
    with m_cl2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">PAM K-Medoids Silhouette</div>
            <div class="metric-value">{kmed_data['silhouette']:.3f}</div>
            <div class="metric-sub">Robust medoid separation</div>
        </div>
        """, unsafe_allow_html=True)
    with m_cl3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">K-Means Davies-Bouldin</div>
            <div class="metric-value">{km_data['davies_bouldin']:.3f}</div>
            <div class="metric-sub">Lower is better</div>
        </div>
        """, unsafe_allow_html=True)
    with m_cl4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">K-Medoids Davies-Bouldin</div>
            <div class="metric-value">{kmed_data['davies_bouldin']:.3f}</div>
            <div class="metric-sub">Lower is better</div>
        </div>
        """, unsafe_allow_html=True)

    # 2D PCA Projections
    p_cl1, p_cl2 = st.columns(2)
    with p_cl1:
        st.markdown("#### K-Means (2D PCA Space with Centroids)")
        fig_km = plot_clustering_pca(clust_res, method="kmeans")
        st.pyplot(fig_km)

    with p_cl2:
        st.markdown("#### K-Medoids PAM (2D PCA Space with Exemplar Medoids)")
        fig_kmed = plot_clustering_pca(clust_res, method="kmedoids")
        st.pyplot(fig_kmed)

    # Cluster Profiles and Personas
    st.markdown("### 👥 Learner Personas & Cluster Profiles")
    for cluster_id, persona in clust_res["personas"].items():
        st.markdown(f"- **Cluster {cluster_id}:** {persona}")

    st.markdown("#### K-Means Cluster Centroid Means (Unscaled)")
    st.dataframe(km_data["profile"], use_container_width=True)

    st.markdown("#### K-Medoids PAM Cluster Medoid Means (Unscaled)")
    st.dataframe(kmed_data["profile"], use_container_width=True)

    # Download Cluster Assignments
    cl_export = km_data["profile"].copy()
    csv_bytes_cl = cl_export.to_csv().encode("utf-8")
    st.download_button(
        label="📥 Download Cluster Profiles (CSV)",
        data=csv_bytes_cl,
        file_name="student_clusters_profile.csv",
        mime="text/csv"
    )


# ==============================================================================
# SECTION 6: EM MISSING VALUE IMPUTATION (EXP 5)
# ==============================================================================
elif "6. EM Missing" in selected_nav:
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem;">🔄 Experiment 5: Expectation-Maximization (EM) Imputation</h1>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Model-based missing value estimation using Gaussian Mixture Models (GMM) with mathematical comparison against baseline median imputation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Explanation Box
    st.markdown("""
    <div class="glass-container">
        <h3 style="margin-top: 0; font-size: 1.15rem; color: #38bdf8;">📐 Mathematical Foundation of EM Imputation</h3>
        <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6;">
            <b>Expectation-Maximization (EM)</b> is an iterative algorithm for maximum likelihood estimation in the presence of latent variables or missing data. 
            Under a Gaussian Mixture Model (GMM) with $K$ components:
        </p>
        <ul style="color: #94a3b8; font-size: 0.85rem; line-height: 1.6;">
            <li><b>Initialization:</b> Components are initialized with k-means; missing values warm-started with medians.</li>
            <li><b>E-step (Expectation):</b> Calculate posterior responsibilities $\\gamma_{ik} = P(Z_i = k \\mid x_i, \\theta)$.</li>
            <li><b>M-step (Maximization):</b> Update component weights $\\pi_k$, means $\\mu_k$, and covariance matrices $\\Sigma_k$.</li>
            <li><b>Conditional Expectation Imputation:</b> Given observed variables $x_{obs}$, the missing entry is estimated via:
                $$\\mathbb{E}[x_{miss} \\mid x_{obs}] = \\sum_{k=1}^K \\gamma_{ik} \\left( \\mu_{k,miss} + \\Sigma_{k,miss,obs} \\Sigma_{k,obs,obs}^{-1} (x_{obs} - \\mu_{k,obs}) \\right)$$
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    em_c1, em_c2, em_c3 = st.columns(3)
    with em_c1:
        target_feat = st.selectbox(
            "Target Feature to Impute",
            ["early_avg_score", "early_vle_clicks", "early_active_days"],
            index=0
        )
    with em_c2:
        missing_rate = st.slider("Synthetic Missingness Rate", 0.05, 0.40, 0.20, 0.05,
                                help="Controlled mask percentage for empirical MAE/RMSE calculation.")
    with em_c3:
        n_comps = st.slider("GMM Mixture Components (K)", 2, 5, 3)

    with st.spinner("Running EM-GMM Imputation Experiment..."):
        em_res = run_em_imputation_experiment(
            df_feat,
            target_feature=target_feat,
            missing_rate=missing_rate,
            n_components=n_comps,
            sample_size=3000
        )

    # Imputation Metrics Comparison
    st.markdown("### 📊 Reconstruction Error Comparison (Ground Truth vs Imputed)")
    em_m1, em_m2, em_m3, em_m4 = st.columns(4)
    with em_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">EM-GMM MAE</div>
            <div class="metric-value" style="color: #38bdf8;">{em_res['em_metrics']['MAE']}</div>
            <div class="metric-sub">Mean Absolute Error</div>
        </div>
        """, unsafe_allow_html=True)
    with em_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Median Imputation MAE</div>
            <div class="metric-value" style="color: #f59e0b;">{em_res['median_metrics']['MAE']}</div>
            <div class="metric-sub">Mean Absolute Error</div>
        </div>
        """, unsafe_allow_html=True)
    with em_m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">EM-GMM RMSE</div>
            <div class="metric-value" style="color: #38bdf8;">{em_res['em_metrics']['RMSE']}</div>
            <div class="metric-sub">Root Mean Squared Error</div>
        </div>
        """, unsafe_allow_html=True)
    with em_m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Median Imputation RMSE</div>
            <div class="metric-value" style="color: #f59e0b;">{em_res['median_metrics']['RMSE']}</div>
            <div class="metric-sub">Root Mean Squared Error</div>
        </div>
        """, unsafe_allow_html=True)

    # Distribution Comparison Plot
    st.markdown("#### Distribution Preservation Comparison")
    fig_em_dist = plot_em_imputation_comparison(
        em_res["true_values"],
        em_res["median_values"],
        em_res["em_values"],
        feature_name=target_feat
    )
    st.pyplot(fig_em_dist)

    # Summary Statistics Table
    st.markdown("#### Descriptive Statistics Before and After Imputation")
    st.dataframe(em_res["stats_df"], use_container_width=True)

    # Sample comparison
    st.markdown("#### Sample Row-by-Row Imputation Estimates")
    st.dataframe(em_res["comparison_sample"], use_container_width=True)


# ==============================================================================
# SECTION 7: HMM WEEKLY ENGAGEMENT ANALYSIS (EXP 4)
# ==============================================================================
elif "7. HMM Weekly" in selected_nav:
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem;">⏱️ Experiment 4: Hidden Markov Model (HMM) Engagement Analysis</h1>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Analyzing temporal student learning engagement dynamics over 36 weeks using the <b>Forward Algorithm</b> (likelihood calculation) 
            and <b>Viterbi Algorithm</b> (hidden state decoding).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Methodology clarification callout
    st.markdown("""
    <div class="glass-container" style="border-left: 4px solid #38bdf8;">
        <h3 style="margin-top: 0; font-size: 1.05rem; color: #38bdf8;">ℹ️ HMM Model Architecture & Clarification</h3>
        <p style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.6;">
            <b>Observations (Discrete):</b> Categorized weekly VLE clicks: 
            <code>No Activity (0)</code>, <code>Low (1-25)</code>, <code>Medium (26-100)</code>, <code>High (>100)</code>.<br>
            <b>Hidden States:</b> Latent engagement levels: 
            <code>0: Inactive</code>, <code>1: Low Engagement</code>, <code>2: Moderate Engagement</code>, <code>3: High Engagement</code>.<br>
            <b>Parameter Notice:</b> As required by experimental guidelines, the baseline state transition matrix $A$ and emission matrix $B$ are 
            <b>illustrative, domain-grounded parameters</b> derived from pedagogical literature (reflecting state persistence and typical drop-off dynamics), 
            ensuring full numerical stability with scaling factors $c_t$ to prevent floating-point underflow.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Student Selector
    st.markdown("### Select Student Record for Temporal Analysis")
    h_c1, h_c2, h_c3 = st.columns(3)
    
    # Available modules
    modules = sorted(df_feat["code_module"].unique())
    with h_c1:
        sel_module = st.selectbox("Course Module", modules, index=0)

    pres_list = sorted(df_feat[df_feat["code_module"] == sel_module]["code_presentation"].unique())
    with h_c2:
        sel_pres = st.selectbox("Presentation", pres_list, index=0)

    stud_subset = df_feat[(df_feat["code_module"] == sel_module) & (df_feat["code_presentation"] == sel_pres)]
    student_ids = stud_subset["id_student"].unique()
    with h_c3:
        sel_student = st.selectbox("Student ID", student_ids[:200], index=0)

    # Extract continuous student sequence
    seq_df = extract_student_sequence(df_weekly, sel_module, sel_pres, sel_student, week_min=0, week_max=35)
    student_meta = stud_subset[stud_subset["id_student"] == sel_student].iloc[0]

    # Run Forward and Viterbi Algorithms
    pi, A, B = get_default_hmm_parameters()
    obs_codes = seq_df["obs_code"].values

    log_likelihood, likelihood, alpha_scaled = forward_algorithm(obs_codes, pi, A, B)
    viterbi_states, v_log_prob = viterbi_algorithm(obs_codes, pi, A, B)

    # Student Summary Metric Cards
    hm1, hm2, hm3, hm4 = st.columns(4)
    with hm1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Final Course Result</div>
            <div class="metric-value" style="font-size: 1.5rem;">{student_meta['final_result']}</div>
            <div class="metric-sub">Risk Target: {'At-Risk' if student_meta['is_at_risk']==1 else 'Not At-Risk'}</div>
        </div>
        """, unsafe_allow_html=True)
    with hm2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Early VLE Clicks</div>
            <div class="metric-value">{student_meta['early_vle_clicks']:,}</div>
            <div class="metric-sub">{student_meta['early_active_days']} Active Days</div>
        </div>
        """, unsafe_allow_html=True)
    with hm3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Forward Log-Likelihood</div>
            <div class="metric-value">{log_likelihood:.2f}</div>
            <div class="metric-sub">log P(O|lambda) via Forward Alg</div>
        </div>
        """, unsafe_allow_html=True)
    with hm4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Viterbi Log Probability</div>
            <div class="metric-value">{v_log_prob:.2f}</div>
            <div class="metric-sub">Optimal path likelihood</div>
        </div>
        """, unsafe_allow_html=True)

    # Timeline Chart
    st.markdown("#### Weekly Clicks and Decoded Latent Engagement States")
    fig_hmm = plot_hmm_engagement_timeline(seq_df, viterbi_states)
    st.pyplot(fig_hmm)

    # Model Matrices Inspection Tab
    with st.expander("🔍 View HMM Transition and Emission Matrices (A & B)"):
        st.markdown("##### State Transition Matrix A: P(State_t | State_{t-1})")
        df_A = pd.DataFrame(A, index=STATE_LABELS.values(), columns=STATE_LABELS.values()).round(3)
        st.dataframe(df_A)

        st.markdown("##### Emission Probability Matrix B: P(Observation | State)")
        df_B = pd.DataFrame(B, index=STATE_LABELS.values(), columns=OBS_LABELS.values()).round(3)
        st.dataframe(df_B)


# ==============================================================================
# SECTION 8: COMPREHENSIVE MODEL COMPARISON
# ==============================================================================
elif "8. Comprehensive" in selected_nav:
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem;">🏆 Comprehensive Model Comparison</h1>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Unified side-by-side benchmarking of all laboratory classification and regression models evaluated on identical test splits.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Benchmark Execution
    st.markdown("### 1. Classification Models Benchmark (Student Risk Prediction)")
    Xc, yc, num_cols_c, cat_cols_c = prepare_classification_dataset(df_feat, include_demographics=True)
    
    # Subsample for fast benchmark
    sub_size = 5000
    indices_c = np.random.RandomState(42).choice(len(Xc), sub_size, replace=False)
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        Xc.iloc[indices_c], yc.iloc[indices_c], test_size=0.25, random_state=42, stratify=yc.iloc[indices_c]
    )

    with st.spinner("Benchmarking Logistic Regression, SVM, KNN, and AdaBoost on identical test partition..."):
        m_lr = train_logistic_regression(X_tr_c, y_tr_c, X_te_c, y_te_c, num_cols_c, cat_cols_c)["metrics"]
        m_svm = train_svm(X_tr_c, y_tr_c, X_te_c, y_te_c, num_cols_c, cat_cols_c, kernel="rbf")["metrics"]
        m_knn = train_knn_classifier(X_tr_c, y_tr_c, X_te_c, y_te_c, num_cols_c, cat_cols_c, k=7)["metrics"]
        m_ada = train_adaboost(X_tr_c, y_tr_c, X_te_c, y_te_c, num_cols_c, cat_cols_c, n_estimators=50)["metrics"]

    clf_summary = pd.DataFrame([
        {"Model": "Logistic Regression (Exp 1)", "Accuracy": m_lr["accuracy"], "Precision": m_lr["precision"], "Recall": m_lr["recall"], "F1-Score": m_lr["f1_score"], "ROC-AUC": m_lr["roc_auc"]},
        {"Model": "Support Vector Machine RBF (Exp 2)", "Accuracy": m_svm["accuracy"], "Precision": m_svm["precision"], "Recall": m_svm["recall"], "F1-Score": m_svm["f1_score"], "ROC-AUC": m_svm["roc_auc"]},
        {"Model": "K-Nearest Neighbors (K=7) (Exp 3)", "Accuracy": m_knn["accuracy"], "Precision": m_knn["precision"], "Recall": m_knn["recall"], "F1-Score": m_knn["f1_score"], "ROC-AUC": m_knn["roc_auc"]},
        {"Model": "AdaBoost Ensemble (Exp 7)", "Accuracy": m_ada["accuracy"], "Precision": m_ada["precision"], "Recall": m_ada["recall"], "F1-Score": m_ada["f1_score"], "ROC-AUC": m_ada["roc_auc"]}
    ]).round(4)

    st.dataframe(clf_summary, use_container_width=True)

    # Comparison Bar Chart
    fig_comp_c, ax_comp_c = plt.subplots(figsize=(8, 3.8))
    fig_comp_c.patch.set_facecolor(CARD_BG)
    ax_comp_c.set_facecolor(CARD_BG)
    for s in ax_comp_c.spines.values():
        s.set_color("#233358")
    ax_comp_c.tick_params(colors="#f1f5f9", labelsize=9)

    x_idx = np.arange(len(clf_summary))
    bar_w = 0.25
    ax_comp_c.bar(x_idx - bar_w, clf_summary["Accuracy"], width=bar_w, color=ACCENT_BLUE, label="Accuracy")
    ax_comp_c.bar(x_idx, clf_summary["F1-Score"], width=bar_w, color=ACCENT_GREEN, label="F1-Score")
    ax_comp_c.bar(x_idx + bar_w, clf_summary["ROC-AUC"], width=bar_w, color=ACCENT_CYAN, label="ROC-AUC")

    ax_comp_c.set_xticks(x_idx)
    ax_comp_c.set_xticklabels(["Logistic", "SVM (RBF)", "KNN (K=7)", "AdaBoost"], fontsize=10, weight="bold")
    ax_comp_c.set_ylabel("Score", color="#f1f5f9", fontsize=10)
    ax_comp_c.set_title("Classification Model Performance Comparison", color="#ffffff", fontsize=11, weight="bold")
    ax_comp_c.legend(facecolor=DARK_BG, edgecolor="#233358", fontsize=9, labelcolor="#f1f5f9")
    plt.tight_layout()
    st.pyplot(fig_comp_c)

    # 2. Regression Benchmark
    st.markdown("### 2. Regression Models Benchmark (Later Assessment Score Prediction)")
    Xr_b, yr_b, num_cols_r_b = prepare_regression_dataset(df_feat)
    indices_r = np.random.RandomState(42).choice(len(Xr_b), sub_size, replace=False)
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        Xr_b.iloc[indices_r], yr_b.iloc[indices_r], test_size=0.25, random_state=42
    )

    with st.spinner("Benchmarking Linear Regression, Polynomial Regression, and KNN Regressor..."):
        m_lin = train_linear_regression(X_tr_r, y_tr_r, X_te_r, y_te_r, num_cols_r_b)["metrics"]
        m_poly = train_polynomial_regression(X_tr_r, y_tr_r, X_te_r, y_te_r, num_cols_r_b, degree=2)["metrics"]
        m_knnr = train_knn_regressor(X_tr_r, y_tr_r, X_te_r, y_te_r, num_cols_r_b, k=7)["metrics"]

    reg_summary = pd.DataFrame([
        {"Model": "Linear Regression (Exp 1)", "MAE": m_lin["mae"], "MSE": m_lin["mse"], "RMSE": m_lin["rmse"], "R2_Score": m_lin["r2"]},
        {"Model": "Polynomial Regression Deg 2 (Exp 1)", "MAE": m_poly["mae"], "MSE": m_poly["mse"], "RMSE": m_poly["rmse"], "R2_Score": m_poly["r2"]},
        {"Model": "KNN Regressor K=7 (Exp 3)", "MAE": m_knnr["mae"], "MSE": m_knnr["mse"], "RMSE": m_knnr["rmse"], "R2_Score": m_knnr["r2"]}
    ]).round(4)

    st.dataframe(reg_summary, use_container_width=True)

    # Download Full Benchmark CSV
    export_text = f"=== CLASSIFICATION BENCHMARK (STUDENT RISK PREDICTION) ===\n{clf_summary.to_csv(index=False)}\n=== REGRESSION BENCHMARK (SUBSEQUENT ASSESSMENT SCORES) ===\n{reg_summary.to_csv(index=False)}"
    csv_bytes_bm = export_text.encode("utf-8")
    st.download_button(
        label="📥 Download Model Benchmark Comparison Table (CSV)",
        data=csv_bytes_bm,
        file_name="model_comparison_benchmark.csv",
        mime="text/csv"
    )


# ==============================================================================
# SECTION 9: ABOUT THE PROJECT & LAB MAPPING
# ==============================================================================
elif "9. About" in selected_nav:
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2rem;">📖 About the Project & Laboratory Mapping</h1>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Academic mapping of Machine Learning Laboratory experiments to implemented Python modules, algorithms, and viva guides.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Mapping Table
    st.markdown("### 📋 Machine Learning Laboratory Experiments Mapping")
    lab_mapping_data = [
        {"Experiment": "Exp 1: Linear, Poly & Logistic Regression", "Algorithm(s)": "LinearRegression, Ridge(Poly deg=2), LogisticRegression", "Module": "src/regression_models.py, src/classification_models.py", "Viva Focus": "Bias-variance tradeoff, odds ratios, L2 regularization"},
        {"Experiment": "Exp 2: Support Vector Machine (SVM)", "Algorithm(s)": "SVC(kernel='rbf'/'linear'/'poly')", "Module": "src/classification_models.py", "Viva Focus": "Kernel trick, margins, support vectors, 2D PCA boundary"},
        {"Experiment": "Exp 3: K-Nearest Neighbors (K-NN)", "Algorithm(s)": "KNeighborsClassifier, KNeighborsRegressor", "Module": "src/classification_models.py, src/regression_models.py", "Viva Focus": "Curse of dimensionality, choice of K, distance metrics"},
        {"Experiment": "Exp 4: Hidden Markov Model (HMM)", "Algorithm(s)": "Forward Algorithm (Likelihood), Viterbi Algorithm (Decoding)", "Module": "src/hmm_model.py", "Viva Focus": "Markov property, trellis diagram, numerical scaling factors"},
        {"Experiment": "Exp 5: Expectation-Maximization (EM)", "Algorithm(s)": "EM-GMM Conditional Expectation Imputer", "Module": "src/em_imputation.py", "Viva Focus": "E-step vs M-step, distribution preservation vs median point mass"},
        {"Experiment": "Exp 6: K-Means & K-Medoids", "Algorithm(s)": "KMeans, PAMKMedoids (pure NumPy PAM implementation)", "Module": "src/clustering_models.py", "Viva Focus": "SSE vs Medoid distance, outlier sensitivity, Silhouette score"},
        {"Experiment": "Exp 7: Ensemble Learning (AdaBoost)", "Algorithm(s)": "AdaBoostClassifier with Decision Stumps", "Module": "src/classification_models.py", "Viva Focus": "Sample weight boosting, weak learner aggregation, margin theory"}
    ]
    st.dataframe(pd.DataFrame(lab_mapping_data), use_container_width=True)

    # Viva Questions and Answers Accordion
    st.markdown("### 🎓 Viva Preparation & Theoretical Justifications")
    with st.expander("Q1: Why is an observation cutoff (Day 60) essential in this learning analytics project?"):
        st.write("""
        An observation cutoff enforces realistic early-warning capability. Without a cutoff, algorithms would experience target leakage 
        by learning from assessments or clicks that happened towards the end of the semester. Using data strictly up to Day 60 replicates 
        a real-world system where academic advisors identify struggling students early enough to deploy tutoring or support interventions.
        """)

    with st.expander("Q2: Why does EM-GMM Imputation outperform simple Median Imputation?"):
        st.write("""
        Median imputation replaces missing values with a single fixed scalar, which unnaturally spikes the distribution at that value, 
        severely deflates sample variance, and destroys multivariate correlations. In contrast, Expectation-Maximization with Gaussian 
        Mixtures estimates the missing coordinates conditional on all observed features via the conditional multivariate Gaussian expectation, 
        preserving the multimodal distribution and cross-feature covariance structure.
        """)

    with st.expander("Q3: How does the Viterbi Algorithm decode the hidden engagement state in HMM?"):
        st.write("""
        The Viterbi algorithm uses dynamic programming in log-probability space to determine the most likely sequence of latent states $Q^* = \\arg\\max_Q P(Q, O \\mid \\lambda)$. 
        At each time step $t$ and state $j$, it computes $v_t(j) = \\max_i [v_{t-1}(i) + \\log A_{ij}] + \\log B_j(o_t)$ and maintains backpointers 
        to trace back the optimal path from $t=T$ to $t=1$.
        """)

    with st.expander("Q4: What is the core difference between K-Means and PAM K-Medoids?"):
        st.write("""
        K-Means calculates cluster centers as the arithmetic mean of points, which minimizes squared Euclidean distance but is highly sensitive to outliers. 
        K-Medoids (PAM) restricts cluster centers to be actual data points (medoids) that minimize the sum of pairwise dissimilarities, making it 
        significantly more robust to extreme outlier behaviors commonly seen in online click logs.
        """)

    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.82rem; margin-top: 30px;">
        Smart Student Performance Prediction & Learning Analytics System • Open University Learning Analytics Dataset (OULAD)
    </div>
    """, unsafe_allow_html=True)
