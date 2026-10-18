# Smart Student Performance Prediction & Learning Analytics System

An end-to-end Machine Learning mini project and interactive Streamlit analytics suite built on the **Open University Learning Analytics Dataset (OULAD)**. This system demonstrates all concepts from the Machine Learning Laboratory curriculum with strict academic rigor, zero target leakage, and reproducible pipelines.

https://smart-student-performance-prediction-and-learning-analytics-sy.streamlit.app/

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Laboratory Experiments Mapping](#-laboratory-experiments-mapping)
3. [Empirical Results & Benchmarks](#-empirical-results--benchmarks)
4. [Project Structure](#-project-structure)
5. [Setup & Quick Start (Windows PowerShell)](#-setup--quick-start-windows-powershell)
6. [Data Preprocessing & Academic Integrity](#-data-preprocessing--academic-integrity)
7. [Viva Questions & Key Justifications](#-viva-questions--key-justifications)
8. [Limitations & Future Enhancements](#-limitations--future-enhancements)

---

## 🌟 Project Overview

The objective of this project is to provide higher education institutions with an evidence-based early warning system that predicts student academic risk before it is too late to intervene.

Using student interactions with the Virtual Learning Environment (VLE), assessment submission dynamics, and demographic profiles:
- **Binary Early Risk Classification:** Identifies whether a student is at risk of failing or withdrawing (`Fail` / `Withdrawn`) vs safe (`Pass` / `Distinction`).
- **Subsequent Score Regression:** Predicts the expected average score on later assessments submitted after day 60.
- **Unsupervised Learner Segmentation:** Groups students into behavioral personas using K-Means and Partitioning Around Medoids (PAM) K-Medoids.
- **Temporal Engagement Modeling:** Tracks weekly student engagement transitions over 36 weeks using a Hidden Markov Model (HMM) with the Forward and Viterbi algorithms.
- **Model-Based Missing Value Imputation:** Compares Expectation-Maximization Gaussian Mixture Model (EM-GMM) imputation against standard median imputation.
- **Ensemble Benchmarking:** Compares AdaBoost with decision stumps against Logistic Regression, Support Vector Machines (SVM), and K-Nearest Neighbors (KNN).

---

## 🔬 Laboratory Experiments Mapping

| Lab Experiment | Implemented Algorithm(s) | Source File | Description & Core Focus |
| :--- | :--- | :--- | :--- |
| **Experiment 1** | Multiple Linear Regression, Polynomial Regression (deg 2), Logistic Regression | [`src/regression_models.py`](file:///D:/ML%20mini%20Project/src/regression_models.py), [`src/classification_models.py`](file:///D:/ML%20mini%20Project/src/classification_models.py) | Regression for later scores (MAE, MSE, RMSE, R²); Logistic Regression for binary risk with odds ratios. |
| **Experiment 2** | Support Vector Machine (SVM) | [`src/classification_models.py`](file:///D:/ML%20mini%20Project/src/classification_models.py) | Compares RBF, Linear, and Polynomial kernels; generates 2D PCA decision boundary contour plots. |
| **Experiment 3** | K-Nearest Neighbors (K-NN) Classifier & Regressor | [`src/classification_models.py`](file:///D:/ML%20mini%20Project/src/classification_models.py), [`src/regression_models.py`](file:///D:/ML%20mini%20Project/src/regression_models.py) | Evaluates optimal $K$ across $K \in [1, 20]$ with safety bounds ($K \le N_{train}$). |
| **Experiment 4** | Hidden Markov Model (HMM) | [`src/hmm_model.py`](file:///D:/ML%20mini%20Project/src/hmm_model.py) | 4 discrete weekly click observations, 4 hidden states; Forward Algorithm (likelihood with scaling) & Viterbi Algorithm (decoding). |
| **Experiment 5** | Expectation-Maximization (EM) Imputation | [`src/em_imputation.py`](file:///D:/ML%20mini%20Project/src/em_imputation.py) | GMM-based conditional expectation imputation vs median baseline; distribution and covariance preservation. |
| **Experiment 6** | K-Means & K-Medoids (PAM) | [`src/clustering_models.py`](file:///D:/ML%20mini%20Project/src/clustering_models.py) | Unsupervised segmentation (no labels); pure NumPy PAM implementation; Silhouette, Davies-Bouldin, and 2D PCA. |
| **Experiment 7** | Ensemble Learning (AdaBoost) | [`src/classification_models.py`](file:///D:/ML%20mini%20Project/src/classification_models.py) | Boosting with Decision Stumps (depth 1); feature importances; compared directly on identical test splits. |

---

## 📊 Empirical Results & Benchmarks

The models were evaluated on the OULAD dataset using an observation cutoff at **Day 60**:

### 1. Classification Models (Student Risk: At-Risk vs Pass/Distinction)
*Evaluated on identical test partition (stratified 25% holdout)*

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AdaBoost Ensemble (Exp 7)** | **76.25%** | **74.12%** | **70.59%** | **0.723** | **0.825** |
| **Support Vector Machine RBF (Exp 2)** | 76.75% | 73.80% | 72.73% | 0.733 | 0.821 |
| **K-Nearest Neighbors (K=7) (Exp 3)** | 76.25% | 72.84% | 73.26% | 0.730 | 0.814 |
| **Logistic Regression (Exp 1)** | 74.25% | 68.82% | 76.47% | 0.725 | 0.809 |

### 2. Regression Models (Subsequent Assessment Score Proxy)
*Evaluated on students with assessments submitted after Day 60*

| Model | MAE | MSE | RMSE | R² Score |
| :--- | :---: | :---: | :---: | :---: |
| **Multiple Linear Regression (Exp 1)** | **7.42** | **91.95** | **9.59** | **0.499** |
| **Polynomial Regression Degree 2 (Exp 1)** | 7.68 | 99.31 | 9.97 | 0.458 |
| **KNN Regressor (K=7) (Exp 3)** | 7.91 | 107.31 | 10.36 | 0.415 |

### 3. Missing Value Estimation (EM-GMM vs Median Imputation)
*Evaluated under 20% controlled synthetic missingness on student scores*

| Method | MAE | RMSE | Variance / Distribution Behavior |
| :--- | :---: | :---: | :--- |
| **EM-GMM Imputation** | **13.97** | **17.84** | Preserves multimodal shape, realistic variance, and multivariate feature covariances. |
| **Baseline Median Imputation** | 14.37 | 18.52 | Creates an artificial point mass at the median, deflating sample variance and distorting correlation. |

### 4. Clustering (K-Means vs PAM K-Medoids)
*Feature set: early VLE clicks, active days, unique sites, early assessment score, studied credits*

| Metric | K-Means (K=3) | PAM K-Medoids (K=3) | Interpretation |
| :--- | :---: | :---: | :--- |
| **Silhouette Score** | **0.203** | **0.190** | Clean cluster separation; medoids are actual representative student data points. |
| **Davies-Bouldin Index** | **1.621** | **1.704** | Lower value indicates compact, well-separated clusters. |

---

## 📁 Project Structure

```
D:\ML mini Project\
├── app.py                     # Interactive Streamlit Web Dashboard (All 9 sections)
├── requirements.txt           # Project dependencies
├── README.md                  # Complete documentation and viva guide
├── data\                      # OULAD raw CSV files (untouched)
│   ├── studentInfo.csv
│   ├── studentAssessment.csv
│   ├── assessments.csv
│   ├── studentRegistration.csv
│   ├── courses.csv
│   ├── studentVle.csv
│   └── vle.csv
├── src\                       # Modular Python packages
│   ├── __init__.py
│   ├── data_loader.py         # Safe CSV loading and schema inspection
│   ├── preprocessing.py       # Day-60 cutoff feature pipeline and caching
│   ├── classification_models.py # Logistic, SVM, KNN, AdaBoost, 2D boundary
│   ├── regression_models.py   # Linear, Polynomial, KNN Regressor
│   ├── clustering_models.py   # Pure NumPy PAM K-Medoids & K-Means
│   ├── hmm_model.py           # Forward Algorithm & Viterbi Algorithm
│   ├── em_imputation.py       # EM-GMM Conditional Expectation Imputer
│   ├── evaluation.py          # Unified metrics and edge-case handling
│   └── utils.py               # Dark-theme Matplotlib/Plotly visualization helpers
└── outputs\                   # Directory for exported predictions and reports
```

---

## 🚀 Setup & Quick Start (Windows PowerShell)

### 1. Open Windows PowerShell in Project Directory
```powershell
cd "D:\ML mini Project"
```

### 2. Verify Python Environment & Dependencies
```powershell
python --version
pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```

The application will launch automatically in your browser at:
`http://localhost:8501`

### 4. Stopping the Dashboard
In the PowerShell terminal running Streamlit, press:
```powershell
Ctrl + C
```

---

## 🛡️ Data Preprocessing & Academic Integrity

### 1. Day 60 Observation Cutoff
In predictive learning analytics, using end-of-term features to predict midterm outcomes causes **target leakage**. This project strictly enforces an observation cutoff at **Day 60**:
- **Inputs ($\le 60$ days):** Early VLE clicks, active days, unique sites visited, early assessment count, early assessment average, early submission delay, registration date, studied credits, previous attempts, and demographics.
- **Classification Target:** `is_at_risk` = 1 if `final_result` in `['Fail', 'Withdrawn']`, else 0 (`['Pass', 'Distinction']`).
- **Regression Target ($> 60$ days):** Average score on assessments submitted after day 60.

### 2. Memory-Safe Chunking
The raw `studentVle.csv` file contains **10,655,280 rows** (~450 MB). To ensure fast execution on standard laptops:
- `studentVle.csv` is read in memory-safe chunks of 1.5 million rows.
- Intermediate duplicate dates and sites are dropped per chunk before computing exact sums.
- Processed feature matrices and weekly click aggregates are cached to `data/cached_student_features_cutoff_60.csv` and `data/cached_student_weekly_vle.csv`, enabling the dashboard to launch in under **0.5 seconds** on subsequent runs!

### 3. String Type Safety
Safe numeric parsing (`pd.to_numeric(..., errors='coerce')`) is applied across all numerical columns (`score`, `date_submitted`, `date`, `sum_click`, `weight`), permanently preventing the Pandas runtime error:
`dtype 'str' does not support operation 'mean'`.

---

## 🎓 Viva Questions & Key Justifications

1. **Why does EM-GMM Imputation preserve distributions better than Median Imputation?**
   Median imputation sets every missing cell to a constant point mass, creating an unnatural spike that destroys feature variance and correlations. EM-GMM models the data as a mixture of Gaussians and calculates the missing value as the conditional expectation given all observed features, accurately preserving multimodal distributions and covariances.

2. **How does the Forward Algorithm avoid floating-point underflow in HMM?**
   Multiplying small transition and emission probabilities over 36 weeks causes probabilities to vanish below double-precision float limits. The implemented Forward Algorithm computes scaling factors $c_t = \frac{1}{\sum_i \alpha_t(i)}$ at every time step, normalizing $\alpha_t$ and accumulating log-likelihood via $\log P(O \mid \lambda) = -\sum_t \log c_t$.

3. **What is the difference between K-Means and PAM K-Medoids?**
   K-Means centers are synthetic means of points (minimizing squared Euclidean distances), which are sensitive to outliers. PAM K-Medoids restricts cluster centers to actual observed student exemplars (medoids) that minimize total absolute dissimilarities, providing superior robustness to online interaction outliers.

4. **Why are Decision Stumps used as weak learners in AdaBoost?**
   Decision stumps (depth-1 decision trees) have high bias and low variance. AdaBoost focuses successive stumps on previously misclassified student records by increasing sample weights, progressively reducing bias while resisting overfitting.

---

## ⚠️ Limitations & Future Enhancements
- **Proxy Regression Target:** Later assessment score is an academic performance proxy based on available mid-term/late assignments, not an official standardized final exam mark.
- **HMM Parameter Baseline:** The default HMM transition ($A$) and emission ($B$) matrices are domain-grounded pedagogical models; fitting via Baum-Welch on small student subsets is supported for comparison.
- **K-Medoids Scalability:** Pairwise distance matrices in PAM scale quadratically ($O(N^2)$); subsampling (e.g. 2,000–5,000 students) is utilized in the UI for instant interactivity.
