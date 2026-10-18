"""
Hidden Markov Model (HMM) for Weekly Student Engagement Analysis.
Implements:
- Forward Algorithm with scaling normalization for observation likelihood
- Viterbi Algorithm in log-space for most likely latent engagement state decoding
- Categorization of weekly clicks into discrete activity levels
- Clean distinction between illustrative pedagogical parameters and Baum-Welch estimated parameters
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional


OBS_LABELS = {
    0: "No Activity",
    1: "Low Activity",
    2: "Medium Activity",
    3: "High Activity"
}

STATE_LABELS = {
    0: "Inactive",
    1: "Low Engagement",
    2: "Moderate Engagement",
    3: "High Engagement"
}

STATE_COLORS = {
    0: "#ef4444",  # Red
    1: "#f59e0b",  # Amber
    2: "#3b82f6",  # Blue
    3: "#10b981"   # Emerald
}


def discretize_clicks(clicks: float) -> int:
    """
    Categorizes raw weekly VLE clicks into 4 discrete observation levels.
    """
    if clicks <= 0:
        return 0  # No activity
    elif clicks <= 25:
        return 1  # Low activity
    elif clicks <= 100:
        return 2  # Medium activity
    else:
        return 3  # High activity


def get_default_hmm_parameters() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns illustrative, domain-grounded HMM parameters reflecting pedagogical dynamics.
    
    States:
    0: Inactive
    1: Low Engagement
    2: Moderate Engagement
    3: High Engagement
    
    NOTE: These parameters are based on educational literature and domain expertise,
    not directly fitted from Baum-Welch.
    """
    # Initial state distribution (at course commencement)
    pi = np.array([0.15, 0.35, 0.35, 0.15], dtype=float)

    # State transition probability matrix A (rows=from, cols=to)
    # High persistence on the diagonal with realistic student state transitions
    A = np.array([
        [0.75, 0.15, 0.08, 0.02],  # Inactive -> mostly stays inactive
        [0.20, 0.50, 0.25, 0.05],  # Low -> can drop or climb
        [0.05, 0.20, 0.55, 0.20],  # Moderate -> steady engagement
        [0.02, 0.08, 0.30, 0.60]   # High -> stays high or drops to moderate
    ], dtype=float)

    # Emission probability matrix B (rows=hidden state, cols=observed activity level)
    # P(Obs | State)
    B = np.array([
        [0.85, 0.12, 0.03, 0.00],  # Inactive: mostly 0 clicks, rarely low
        [0.30, 0.50, 0.18, 0.02],  # Low engagement: mostly low or 0 clicks
        [0.05, 0.25, 0.55, 0.15],  # Moderate: mostly medium clicks
        [0.01, 0.05, 0.34, 0.60]   # High engagement: mostly high clicks
    ], dtype=float)

    return pi, A, B


def forward_algorithm(
    obs_seq: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    B: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    """
    Implements the Forward Algorithm with scaling coefficients (ct)
    to calculate sequence likelihood P(O|lambda) without floating-point underflow.
    
    Returns:
    - log_likelihood : float (log P(O|lambda))
    - likelihood : float (exp(log_likelihood))
    - alpha_scaled : np.ndarray of shape (T, N)
    """
    T = len(obs_seq)
    N = len(pi)

    alpha = np.zeros((T, N), dtype=float)
    c = np.zeros(T, dtype=float)

    # Step 1: Initialization at t = 0
    o0 = int(obs_seq[0])
    alpha[0, :] = pi * B[:, o0]
    c[0] = 1.0 / (np.sum(alpha[0, :]) + 1e-15)
    alpha[0, :] *= c[0]

    # Step 2: Induction for t = 1 ... T-1
    for t in range(1, T):
        ot = int(obs_seq[t])
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t - 1, :] * A[:, j]) * B[j, ot]
        c[t] = 1.0 / (np.sum(alpha[t, :]) + 1e-15)
        alpha[t, :] *= c[t]

    # Step 3: Termination
    log_likelihood = -float(np.sum(np.log(c + 1e-15)))
    likelihood = float(np.exp(np.clip(log_likelihood, -700, 0)))

    return log_likelihood, likelihood, alpha


def viterbi_algorithm(
    obs_seq: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    B: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Implements the Viterbi Algorithm in log-domain for numerical stability
    to estimate the most likely hidden engagement state sequence Q*.
    
    Returns:
    - best_path : np.ndarray of shape (T,) with state indices 0..N-1
    - best_log_prob : float
    """
    T = len(obs_seq)
    N = len(pi)

    # Convert to log-probabilities with epsilon safety against log(0)
    eps = 1e-12
    log_pi = np.log(np.maximum(pi, eps))
    log_A = np.log(np.maximum(A, eps))
    log_B = np.log(np.maximum(B, eps))

    viterbi = np.zeros((T, N), dtype=float)
    backpointer = np.zeros((T, N), dtype=int)

    # Step 1: Initialization
    o0 = int(obs_seq[0])
    viterbi[0, :] = log_pi + log_B[:, o0]
    backpointer[0, :] = 0

    # Step 2: Recursion
    for t in range(1, T):
        ot = int(obs_seq[t])
        for j in range(N):
            # v_t(j) = max_i [ v_{t-1}(i) + log A_ij ] + log B_j(o_t)
            trans_probs = viterbi[t - 1, :] + log_A[:, j]
            best_prev = np.argmax(trans_probs)
            viterbi[t, j] = trans_probs[best_prev] + log_B[j, ot]
            backpointer[t, j] = best_prev

    # Step 3: Termination
    best_log_prob = float(np.max(viterbi[T - 1, :]))
    best_last_state = int(np.argmax(viterbi[T - 1, :]))

    # Step 4: Backtracking
    best_path = np.zeros(T, dtype=int)
    best_path[T - 1] = best_last_state
    for t in range(T - 2, -1, -1):
        best_path[t] = backpointer[t + 1, best_path[t + 1]]

    return best_path, best_log_prob


def extract_student_sequence(
    weekly_vle_df: pd.DataFrame,
    code_module: str,
    code_presentation: str,
    id_student: int,
    week_min: int = 0,
    week_max: int = 35
) -> pd.DataFrame:
    """
    Extracts complete continuous weekly click series for a specific student,
    filling non-active weeks with 0 clicks.
    """
    sub = weekly_vle_df[
        (weekly_vle_df["code_module"] == code_module) &
        (weekly_vle_df["code_presentation"] == code_presentation) &
        (weekly_vle_df["id_student"] == id_student)
    ].copy()

    weeks_all = pd.DataFrame({"week": range(week_min, week_max + 1)})
    merged = weeks_all.merge(sub[["week", "sum_click"]], on="week", how="left")
    merged["sum_click"] = merged["sum_click"].fillna(0).astype(int)
    merged["obs_code"] = merged["sum_click"].apply(discretize_clicks)
    merged["obs_label"] = merged["obs_code"].map(OBS_LABELS)

    return merged
