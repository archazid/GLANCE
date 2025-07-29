# scripts/create_boxplots.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
PROJECT_ROOT = "C:/Users/Archazid/projects/GLANCE/"
DATA_FILE = os.path.join(PROJECT_ROOT, "exps/results/plotting_data_combined.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "exps/results/performance_boxplots.png")

# Define the order you want the models to appear on the x-axis
MODEL_ORDER = [
    "GLANCE-LR",
    "GLANCE+File",
    "GLANCE+Line",
    "GLANCE+",
]

# Define which metrics to plot and their y-axis labels
METRICS_TO_PLOT = {
    "recall": "Recall (Higher is Better)",
    "far": "FAR (Lower is Better)",
    "ce": "CoF (Lower is Better)",  # 'ce' is CoF
    "d2h": "D2H (Lower is Better)",
    "mcc": "MCC (Higher is Better)",
    "ifa": "IFA (Lower is Better)",
    "recall_20": "Recall@20%LOC (Higher is Better)",
    "ratio": "HoB (Higher is Better)",  # 'ratio' is HoB
}

# --- PLOTTING LOGIC ---

# Load the prepared data
df = pd.read_csv(DATA_FILE)

# Set the style for the plots (e.g., 'whitegrid' is a clean style)
sns.set_theme(style="whitegrid")

# Create a 2x4 figure for our subplots
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle(
    "Comprehensive Performance Comparison of GLANCE Model Variants", fontsize=24, y=0.97
)

# Flatten the 2x4 axes array into a 1D array for easy iteration
axes = axes.flatten()

# Generate one boxplot for each metric
for i, (metric, y_label) in enumerate(METRICS_TO_PLOT.items()):
    ax = axes[i]

    # Create the boxplot using seaborn
    sns.boxplot(
        x="model",
        y=metric,
        data=df,
        order=MODEL_ORDER,
        ax=ax,
        palette="viridis",
    )

    # Customize labels and titles for each subplot
    ax.set_title(y_label, fontsize=16)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.tick_params(axis="x", labelrotation=20, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

# Adjust layout to prevent titles and labels from overlapping
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the final figure to a file
plt.savefig(OUTPUT_FILE, dpi=300)
print(f"Boxplot figure saved to {OUTPUT_FILE}")

# Optionally, display the plot
plt.show()
