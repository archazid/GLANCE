# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np


# --- IMPORTANT NOTE FOR YOUR THESIS ---
# This file provides a TEMPLATE for extracting hybrid features.
# The function `extract_hybrid_features` currently generates MOCK data.

# Note: Will be implemented on Phase 2 of my research plan
def extract_hybrid_features(release_name: str, file_list: list) -> pd.DataFrame:
    """
    Extracts a hybrid set of features for a given list of files in a release.

    This function should be implemented to extract:
    1. Process Metrics (from Git history)
    2. Complexity Metrics (from static code analysis)
    3. CK Metrics (from static code analysis)

    Args:
        release_name (str): The name of the software release (e.g., 'ambari-1.2.0').
        file_list (list): A list of file paths relative to the project root.

    Returns:
        pd.DataFrame: A DataFrame where the index is the filename and columns are the extracted features.
    """
    print(f"Extracting hybrid features for {len(file_list)} files in {release_name}...")

    # --- REAL IMPLEMENTATION GUIDE ---
    # Use proper libraries here to get the real data.
    #
    # 1. For Process Metrics (Code Churn, etc.):
    #    - Use the 'pydriller' library (pip install pydriller).
    #    - For each file in file_list, iterate through its commit history
    #      to calculate revisions, authors, age, etc.
    #    - Example: `from pydriller import Repository`
    #               `for commit in Repository('path/to/your/repo', to_commit='tag_of_release').traverse_commits():`
    #               `    for modified_file in commit.modified_files:`
    #               `        if modified_file.new_path in file_list: ...`
    #
    # 2. For Complexity and CK Metrics:
    #    - For Java, use command-line tools like 'CK' (https://github.com/mauricioaniche/ck)
    #      or 'MetricsReloaded' (https://github.com/MetricsReloaded/MetricsReloaded).
    #    - Run this tool on the codebase of the specific release and parse its CSV output.
    #
    # --- MOCK IMPLEMENTATION (for demonstration and testing) ---
    # Generate random data to make this script runnable without the full toolchain.
    # Replace this section with the real implementation.

    num_files = len(file_list)
    mock_data = {
        "filename": file_list,
        "revisions": np.random.randint(1, 50, size=num_files),
        "authors": np.random.randint(1, 10, size=num_files),
        "age_weeks": np.random.randint(10, 200, size=num_files),
        "loc": np.random.randint(20, 1000, size=num_files),
        "cyclomatic_complexity": np.random.randint(1, 20, size=num_files),
        "cbo": np.random.randint(0, 30, size=num_files),
        "wmc": np.random.randint(5, 100, size=num_files),
        "lcom": np.random.uniform(0, 2, size=num_files),
    }

    feature_df = pd.DataFrame(mock_data)
    feature_df.set_index("filename", inplace=True)

    print("Finished extracting features.")
    return feature_df
