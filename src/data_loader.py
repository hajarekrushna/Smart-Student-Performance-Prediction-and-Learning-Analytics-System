"""
Data Loader Module for Open University Learning Analytics Dataset (OULAD).
Locates CSV files, performs safe loading, and caches processed summaries.
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Tuple

REQUIRED_OULAD_FILES = [
    "studentInfo.csv",
    "studentAssessment.csv",
    "assessments.csv",
    "studentRegistration.csv",
    "courses.csv",
    "studentVle.csv"
]

OPTIONAL_OULAD_FILES = [
    "vle.csv"
]


def find_data_dir(search_root: Optional[str] = None) -> str:
    """
    Automatically search for the directory containing the OULAD CSV files.
    Handles nested folders or direct paths.
    """
    candidate_roots = []
    if search_root:
        candidate_roots.append(search_root)
        candidate_roots.append(os.path.join(search_root, "data"))

    # Also check current working directory and relative data
    cwd = os.getcwd()
    candidate_roots.extend([
        os.path.join(cwd, "data"),
        cwd,
        r"D:\ML mini Project\data",
        r"D:\ML mini Project"
    ])

    for root in candidate_roots:
        if os.path.isdir(root):
            # Check if cached features or raw tables are present
            has_cache = os.path.isfile(os.path.join(root, "cached_student_features_cutoff_60.csv"))
            present = [f for f in REQUIRED_OULAD_FILES if os.path.isfile(os.path.join(root, f))]
            if has_cache or len(present) >= 3:
                return os.path.abspath(root)

            # Check immediate subdirectories for nested extractions
            try:
                for sub in os.listdir(root):
                    sub_path = os.path.join(root, sub)
                    if os.path.isdir(sub_path):
                        has_cache_sub = os.path.isfile(os.path.join(sub_path, "cached_student_features_cutoff_60.csv"))
                        present_sub = [f for f in REQUIRED_OULAD_FILES if os.path.isfile(os.path.join(sub_path, f))]
                        if has_cache_sub or len(present_sub) >= 3:
                            return os.path.abspath(sub_path)
            except Exception:
                pass

    raise FileNotFoundError(
        f"Could not locate OULAD CSV files. Searched paths: {candidate_roots}. "
        f"Expected files: {REQUIRED_OULAD_FILES}"
    )


def load_raw_csv(data_dir: str, filename: str, usecols: Optional[List[str]] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Safely load a raw CSV file handling '?' missing value strings and safe string parsing.
    """
    filepath = os.path.join(data_dir, filename)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File '{filename}' not found in '{data_dir}'.")

    # Safe read with '?' treated as NaN
    df = pd.read_csv(filepath, na_values=["?", "", "NA", "null", "NULL"], usecols=usecols, nrows=nrows)
    return df


def inspect_all_tables(data_dir: str) -> Dict[str, Dict]:
    """
    Inspect all OULAD tables without loading massive data into RAM.
    Returns metadata including row count, columns, missing values, and dtypes.
    """
    summary = {}
    all_files = REQUIRED_OULAD_FILES + OPTIONAL_OULAD_FILES
    
    for filename in all_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            continue
            
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        # Read small sample to get schema
        sample = pd.read_csv(filepath, nrows=100, na_values=["?", "", "NA"])
        
        # Fast line count estimation or exact count
        row_count = 0
        with open(filepath, "rb") as f:
            for _ in f:
                row_count += 1
        row_count = max(0, row_count - 1)  # Subtract header

        summary[filename] = {
            "path": filepath,
            "size_mb": round(file_size_mb, 2),
            "total_rows": row_count,
            "columns": list(sample.columns),
            "sample_head": sample.head(5),
            "dtypes": {col: str(sample[col].dtype) for col in sample.columns}
        }
        
    return summary
