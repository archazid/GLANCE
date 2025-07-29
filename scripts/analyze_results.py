# -*- coding: utf-8 -*-
import sys
import warnings
import pandas as pd
from pandas import DataFrame
import os

# Ensure the project root is in the Python path
# Modify this path to match the location on your machine
PROJECT_ROOT = "C:/Users/Archazid/projects/GLANCE/"
sys.path.append(PROJECT_ROOT)

# Import all models we want to analyze
from src.models.glance import Glance_EA, Glance_LR, Glance_MD
from src.models.glance_plus_file import GlancePlus_File
from src.models.glance_plus_line import GlancePlus_Line_LR
from src.models.glance_plus import GlancePlus

from src.utils.helper import make_path, get_project_releases_dict, get_project_list

# Ignore warning information
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

# Define all models to be included in the analysis
# The key is the desired column name in the output, the value is the model class
MODELS_TO_ANALYZE = {
    # Original Baselines
    "GLANCE-LR": Glance_LR,
    # Enhanced File-Level Classifier Only
    "GLANCE+File": GlancePlus_File,
    # Enhanced Line-Level Metrics Only
    "GLANCE+Line": GlancePlus_Line_LR,
    # Combined Approaches
    "GLANCE+": GlancePlus,
}

# Define the performance indicators to be analyzed
INDICATORS = [
    "precision",
    "recall",
    "far",
    "ce",
    "d2h",
    "mcc",
    "ifa",
    "recall_20",
    "ratio",
]

# Define the output directory for our analysis results
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "exps/results/")
make_path(OUTPUT_PATH)


def get_model_results_df(model_class):
    """
    Reads the raw line-level evaluation results for a given model class.
    """
    # Instantiate the model with dummy values just to get its result path
    model_instance = model_class()
    results_file = model_instance.line_level_evaluation_file

    if not os.path.exists(results_file):
        print(
            f"WARNING: Results file not found for {model_instance.model_name}. Skipping. Path: {results_file}"
        )
        return None

    # Read the raw results, selecting only the indicators we care about
    return pd.read_csv(results_file)[INDICATORS]


def generate_per_project_summary(model_name, model_class):
    """
    Generates a per-project summary (mean performance) for a single model.
    """
    raw_results_df = get_model_results_df(model_class)
    if raw_results_df is None:
        return

    project_summary_data = []
    project_names = get_project_list()
    project_releases_dict = get_project_releases_dict()

    last_row_index = 0
    for project in project_names:
        # Determine the number of test releases for this project
        num_test_releases = len(project_releases_dict[project]) - 1
        start_index = last_row_index
        end_index = last_row_index + num_test_releases

        # Slice the DataFrame to get results for the current project
        project_results = raw_results_df.iloc[start_index:end_index]

        # Calculate the mean performance for this project
        project_summary_data.append(list(project_results.mean(axis=0)))

        last_row_index = end_index

    summary_df = DataFrame(
        project_summary_data, index=project_names, columns=INDICATORS
    )
    summary_df.to_csv(
        os.path.join(OUTPUT_PATH, f"summary_per_project_{model_name}.csv"), index=True
    )
    print(f"Generated per-project summary for {model_name}.")


def generate_overall_comparison_summary():
    """
    Generates a comprehensive summary table comparing the overall performance
    (mean, median, std) of all models across all projects.
    """
    all_models_summary = []

    for model_name, model_class in MODELS_TO_ANALYZE.items():
        raw_results_df = get_model_results_df(model_class)
        if raw_results_df is None:
            continue

        # Calculate statistics across ALL prediction pairs
        mean_perf = raw_results_df.mean()
        median_perf = raw_results_df.median()
        std_perf = raw_results_df.std()

        # Add a multi-level index for clarity in the final table
        mean_perf.name = (model_name, "Mean")
        median_perf.name = (model_name, "Median")
        std_perf.name = (model_name, "Std Dev")

        all_models_summary.extend([mean_perf, median_perf, std_perf])

    if not all_models_summary:
        print("No results found for any model. Exiting.")
        return

    # Combine all stats into a single, beautifully formatted DataFrame
    comparison_df = pd.concat(all_models_summary, axis=1).T

    # Save to CSV
    comparison_df.to_csv(
        os.path.join(OUTPUT_PATH, "summary_overall_comparison.csv"), index=True
    )
    print("\nGenerated overall model comparison summary.")

    # Print a formatted version to the console for quick viewing
    print("\n--- Overall Model Performance Comparison ---")
    print(comparison_df.round(3))
    print("------------------------------------------")


if __name__ == "__main__":
    # --- Generate the per-project summary for each model ---
    # This is useful for detailed, project-by-project analysis.
    print("--- Generating Per-Project Summaries ---")
    for name, model_cls in MODELS_TO_ANALYZE.items():
        generate_per_project_summary(name, model_cls)

    # --- Generate the final, overall comparison table ---
    # This is the key table for your thesis results chapter.
    generate_overall_comparison_summary()

    print("\nAnalysis complete. Results are saved in:", OUTPUT_PATH)
