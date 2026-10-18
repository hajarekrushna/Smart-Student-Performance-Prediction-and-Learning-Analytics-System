"""
Feature Engineering and Preprocessing Pipeline for OULAD.
Extracts student-level features up to a strict observation cutoff (default: day 60)
to prevent target leakage, builds weekly VLE engagement for HMM, and caches results.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Callable


def build_preprocessed_features(
    data_dir: str,
    cutoff_day: int = 60,
    force_recompute: bool = False,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds or loads cached student features and weekly VLE activity.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing raw OULAD CSV files.
    cutoff_day : int
        Observation cutoff day (default 60). Features use only data <= cutoff_day.
    force_recompute : bool
        If True, re-aggregates from raw CSV files even if cache exists.
    progress_callback : callable, optional
        Function (message, fraction) for UI progress updates.
        
    Returns:
    --------
    features_df : pd.DataFrame
        Student-course level features and targets (classification and regression).
    weekly_vle_df : pd.DataFrame
        Weekly VLE click aggregations for HMM engagement modeling.
    """
    cache_features_path = os.path.join(data_dir, f"cached_student_features_cutoff_{cutoff_day}.csv")
    cache_weekly_path = os.path.join(data_dir, "cached_student_weekly_vle.csv")

    if not force_recompute and os.path.isfile(cache_features_path) and os.path.isfile(cache_weekly_path):
        if progress_callback:
            progress_callback("Loading pre-processed data from cache...", 1.0)
        df_feat = pd.read_csv(cache_features_path)
        df_weekly = pd.read_csv(cache_weekly_path)
        return df_feat, df_weekly

    # 1. Load studentInfo
    if progress_callback:
        progress_callback("Reading student demographics and course records...", 0.1)
    
    info_path = os.path.join(data_dir, "studentInfo.csv")
    df_info = pd.read_csv(info_path, na_values=["?", "", "NA"])
    
    # Target definition: At-risk = Fail or Withdrawn (1), Not-at-risk = Pass or Distinction (0)
    df_info["is_at_risk"] = df_info["final_result"].apply(
        lambda r: 1 if str(r).strip() in ["Fail", "Withdrawn"] else 0
    )
    df_info["num_of_prev_attempts"] = pd.to_numeric(df_info["num_of_prev_attempts"], errors="coerce").fillna(0).astype(int)
    df_info["studied_credits"] = pd.to_numeric(df_info["studied_credits"], errors="coerce").fillna(60).astype(int)

    # 2. Load studentRegistration
    if progress_callback:
        progress_callback("Processing registration data...", 0.2)
    reg_path = os.path.join(data_dir, "studentRegistration.csv")
    df_reg = pd.read_csv(reg_path, na_values=["?", "", "NA"])
    df_reg["date_registration"] = pd.to_numeric(df_reg["date_registration"], errors="coerce")
    # Merge registration
    df_merged = df_info.merge(
        df_reg[["code_module", "code_presentation", "id_student", "date_registration"]],
        on=["code_module", "code_presentation", "id_student"],
        how="left"
    )

    # 3. Load & Process Assessments (Early features vs Later target)
    if progress_callback:
        progress_callback("Aggregating assessment submissions before/after cutoff...", 0.35)
    
    assess_path = os.path.join(data_dir, "assessments.csv")
    stud_assess_path = os.path.join(data_dir, "studentAssessment.csv")
    
    df_assess = pd.read_csv(assess_path, na_values=["?", "", "NA"])
    df_stud_assess = pd.read_csv(stud_assess_path, na_values=["?", "", "NA"])

    # Safe type conversions to prevent "dtype 'str' does not support operation 'mean'"
    df_assess["date"] = pd.to_numeric(df_assess["date"], errors="coerce")
    df_assess["weight"] = pd.to_numeric(df_assess["weight"], errors="coerce").fillna(0.0)
    df_stud_assess["score"] = pd.to_numeric(df_stud_assess["score"], errors="coerce")
    df_stud_assess["date_submitted"] = pd.to_numeric(df_stud_assess["date_submitted"], errors="coerce")
    
    # Merge assessments with metadata
    assess_merged = df_stud_assess.merge(df_assess, on="id_assessment", how="inner")
    
    # Submission delay relative to due date (negative = early, positive = late)
    assess_merged["submission_delay"] = assess_merged["date_submitted"] - assess_merged["date"]
    
    # Split into Early (<= cutoff_day) and Later (> cutoff_day)
    early_assess = assess_merged[assess_merged["date_submitted"] <= cutoff_day].copy()
    later_assess = assess_merged[assess_merged["date_submitted"] > cutoff_day].copy()
    
    # Early assessment aggregates per student-course
    early_agg = early_assess.groupby(["code_module", "code_presentation", "id_student"]).agg(
        early_assessment_count=("id_assessment", "count"),
        early_avg_score=("score", "mean"),
        early_avg_delay=("submission_delay", "mean")
    ).reset_index()

    # Later assessment aggregate (Regression Target: proxy for later assessment performance)
    later_agg = later_assess.groupby(["code_module", "code_presentation", "id_student"]).agg(
        later_avg_score=("score", "mean"),
        later_assessment_count=("id_assessment", "count")
    ).reset_index()

    df_merged = df_merged.merge(early_agg, on=["code_module", "code_presentation", "id_student"], how="left")
    df_merged = df_merged.merge(later_agg, on=["code_module", "code_presentation", "id_student"], how="left")
    
    # Fill early assessment count with 0 for students who didn't submit any early assessment
    df_merged["early_assessment_count"] = df_merged["early_assessment_count"].fillna(0).astype(int)
    # Note: early_avg_score and early_avg_delay can be NaN if no assessment was taken, which we handle in pipelines

    # 4. Chunked Processing of studentVle.csv (Early clicks & Weekly clicks)
    if progress_callback:
        progress_callback("Aggregating 10M+ student VLE click records in memory-safe chunks...", 0.55)

    vle_path = os.path.join(data_dir, "studentVle.csv")
    chunk_size = 1500000

    clicks_list = []
    days_sites_list = []
    weekly_list = []

    for chunk in pd.read_csv(
        vle_path,
        chunksize=chunk_size,
        usecols=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
        na_values=["?", "", "NA"]
    ):
        chunk["date"] = pd.to_numeric(chunk["date"], errors="coerce").fillna(0)
        chunk["sum_click"] = pd.to_numeric(chunk["sum_click"], errors="coerce").fillna(0)

        # 4A. Early activity (date <= cutoff_day)
        c_early = chunk[chunk["date"] <= cutoff_day]
        if not c_early.empty:
            c_clicks = c_early.groupby(["code_module", "code_presentation", "id_student"])["sum_click"].sum().reset_index()
            clicks_list.append(c_clicks)

            c_days = c_early[["code_module", "code_presentation", "id_student", "date"]].drop_duplicates()
            c_sites = c_early[["code_module", "code_presentation", "id_student", "id_site"]].drop_duplicates()
            days_sites_list.append((c_days, c_sites))

        # 4B. Weekly activity for HMM (relative to course start, weeks -2 to 38)
        chunk["week"] = (chunk["date"] // 7).astype(int)
        chunk_w = chunk[(chunk["week"] >= -2) & (chunk["week"] <= 38)]
        if not chunk_w.empty:
            w_agg = chunk_w.groupby(["code_module", "code_presentation", "id_student", "week"])["sum_click"].sum().reset_index()
            weekly_list.append(w_agg)

    if progress_callback:
        progress_callback("Finalizing VLE click totals and active day indicators...", 0.75)

    # Consolidate early VLE metrics
    all_clicks = pd.concat(clicks_list).groupby(["code_module", "code_presentation", "id_student"])["sum_click"].sum().reset_index()
    all_clicks.rename(columns={"sum_click": "early_vle_clicks"}, inplace=True)

    all_days = pd.concat([d[0] for d in days_sites_list]).drop_duplicates()
    active_days = all_days.groupby(["code_module", "code_presentation", "id_student"])["date"].count().reset_index()
    active_days.rename(columns={"date": "early_active_days"}, inplace=True)

    all_sites = pd.concat([d[1] for d in days_sites_list]).drop_duplicates()
    unique_sites = all_sites.groupby(["code_module", "code_presentation", "id_student"])["id_site"].count().reset_index()
    unique_sites.rename(columns={"id_site": "early_unique_sites"}, inplace=True)

    # Merge VLE early indicators
    vle_early_combined = all_clicks.merge(active_days, on=["code_module", "code_presentation", "id_student"], how="outer")
    vle_early_combined = vle_early_combined.merge(unique_sites, on=["code_module", "code_presentation", "id_student"], how="outer")

    df_merged = df_merged.merge(vle_early_combined, on=["code_module", "code_presentation", "id_student"], how="left")
    
    # Fill missing VLE activity with 0 (students who never clicked before day 60)
    df_merged["early_vle_clicks"] = df_merged["early_vle_clicks"].fillna(0).astype(int)
    df_merged["early_active_days"] = df_merged["early_active_days"].fillna(0).astype(int)
    df_merged["early_unique_sites"] = df_merged["early_unique_sites"].fillna(0).astype(int)
    df_merged["early_clicks_per_active_day"] = (
        df_merged["early_vle_clicks"] / np.maximum(1, df_merged["early_active_days"])
    ).round(2)

    # 4C. Consolidate weekly activity for HMM
    if progress_callback:
        progress_callback("Consolidating weekly activity for HMM...", 0.9)

    weekly_vle_df = pd.concat(weekly_list).groupby(
        ["code_module", "code_presentation", "id_student", "week"]
    )["sum_click"].sum().reset_index()

    # Save cached files
    df_merged.to_csv(cache_features_path, index=False)
    weekly_vle_df.to_csv(cache_weekly_path, index=False)

    if progress_callback:
        progress_callback("Dataset preprocessing complete and cached!", 1.0)

    return df_merged, weekly_vle_df


def prepare_classification_dataset(
    df: pd.DataFrame,
    include_demographics: bool = True
) -> Tuple[pd.DataFrame, pd.Series, list, list]:
    """
    Prepares feature matrix X and target y for student risk classification.
    Strictly uses information available up to cutoff day (no data leakage).
    """
    numeric_features = [
        "early_vle_clicks",
        "early_active_days",
        "early_unique_sites",
        "early_clicks_per_active_day",
        "early_assessment_count",
        "early_avg_score",
        "early_avg_delay",
        "date_registration",
        "studied_credits",
        "num_of_prev_attempts"
    ]
    
    categorical_features = []
    if include_demographics:
        categorical_features = [
            "gender",
            "region",
            "highest_education",
            "imd_band",
            "age_band",
            "disability",
            "code_module"
        ]

    feature_cols = numeric_features + categorical_features
    X = df[feature_cols].copy()
    y = df["is_at_risk"].astype(int)
    
    return X, y, numeric_features, categorical_features


def prepare_regression_dataset(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Prepares dataset for predicting later assessment score proxy.
    Only includes students who submitted at least one later assessment (> day 60).
    """
    # Only records where later_avg_score is not null
    reg_df = df[df["later_avg_score"].notnull()].copy()
    
    numeric_features = [
        "early_vle_clicks",
        "early_active_days",
        "early_unique_sites",
        "early_clicks_per_active_day",
        "early_assessment_count",
        "early_avg_score",
        "early_avg_delay",
        "date_registration",
        "studied_credits",
        "num_of_prev_attempts"
    ]
    
    X = reg_df[numeric_features].copy()
    y = reg_df["later_avg_score"].copy()
    
    return X, y, numeric_features
