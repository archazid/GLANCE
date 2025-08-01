# scripts/prepare_plot_data.py
import pandas as pd
import os
import sys

# Add the project root to the Python path
PROJECT_ROOT = "C:/Users/Archazid/projects/GLANCE/"
sys.path.append(PROJECT_ROOT)

from src.models.glance import Glance_EA, Glance_LR, Glance_MD
from src.models.glance_plus_file import GlancePlus_File
from src.models.glance_plus_line import GlancePlus_Line_LR
from src.models.glance_plus import GlancePlus

# Define the models you want to include in the plot
MODELS_TO_PLOT = {
    # Original Baselines
    "GLANCE-LR": Glance_LR,
    # Enhanced File-Level Classifier Only
    "GLANCE+File": GlancePlus_File,
    # Enhanced Line-Level Metrics Only
    "GLANCE+Line": GlancePlus_Line_LR,
    # Combined Approaches
    "GLANCE++": GlancePlus,
}


def gather_all_results():
    all_results = []
    for model_name, model_class in MODELS_TO_PLOT.items():
        # Instantiate the model to get the path to its results file
        model_instance = model_class()
        results_file = model_instance.line_level_evaluation_file

        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            # Add a new column to identify the model
            df["model"] = model_name
            all_results.append(df)
        else:
            print(f"Warning: Results file not found for {model_name}. Skipping.")

    # Combine all individual DataFrames into one large DataFrame
    combined_df = pd.concat(all_results, ignore_index=True)

    # Save the combined data to a new CSV for easy access
    output_path = os.path.join(PROJECT_ROOT, "exps/results/")
    combined_df.to_csv(
        os.path.join(output_path, "plotting_data_combined.csv"), index=False
    )
    print(
        f"Combined data saved to {os.path.join(output_path, 'plotting_data_combined.csv')}"
    )


if __name__ == "__main__":
    gather_all_results()
