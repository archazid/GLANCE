# Title: stats.r
# Purpose: Perform statistical analysis for the enhanced GLANCE models.

# --- 1. SETUP ---

# Install required packages if you don't have them yet
# install.packages("effsize")

library(effsize)

# --- 2. CONFIGURATION ---
# UPDATE THESE VARIABLES TO MATCH YOUR PROJECT STRUCTURE

# Set the path to the directory containing all your model result folders
RESULTS_PATH <- "C:/Users/Archazid/projects/GLANCE/Result/"
EXPS_RESULTS_PATH <- "C:/Users/Archazid/projects/GLANCE/exps/results/"

# Define your baseline model (the one all others will be compared against)
BASELINE_MODEL_NAME <- "BASE-Glance-LR"

# Define the list of your new, enhanced models to compare against the baseline
CHALLENGER_MODELS <- c(
  "GlancePlus-File",
  "GlancePlus-Line-LR",
  "GlancePlus"
)

# Define the performance indicators to analyze
# The names must exactly match the column headers in your CSV files
INDICATORS <- c("recall", "far", "ce", "d2h", "mcc", "ifa", "recall_20", "ratio")


# --- 3. HELPER FUNCTIONS ---

# Function to interpret Cliff's Delta magnitude
get_magnitude <- function(d) {
  if (is.na(d)) return("N/A")
  d_abs <- abs(d)
  if (d_abs < 0.147) {
    return("Negligible")
  } else if (d_abs < 0.33) {
    return("Small")
  } else if (d_abs < 0.474) {
    return("Medium")
  } else {
    return("Large")
  }
}


# --- 4. MAIN ANALYSIS SCRIPT ---

# Load the baseline data
baseline_file_path <- paste0(RESULTS_PATH, BASELINE_MODEL_NAME, "/line_result/evaluation.csv")
if (!file.exists(baseline_file_path)) {
  stop("Baseline results file not found: ", baseline_file_path)
}
baseline_data <- read.csv(baseline_file_path, header = TRUE)

# Initialize a list to store the results
results_list <- list()

# Loop through each challenger model and compare it to the baseline
for (challenger_name in CHALLENGER_MODELS) {
  
  challenger_file_path <- paste0(RESULTS_PATH, challenger_name, "/line_result/evaluation.csv")
  if (!file.exists(challenger_file_path)) {
    print(paste("Warning: Results file not found for", challenger_name, ". Skipping."))
    next
  }
  challenger_data <- read.csv(challenger_file_path, header = TRUE)
  
  # Loop through each performance indicator
  for (indicator in INDICATORS) {
    
    # Perform the Wilcoxon signed-rank test
    p_value <- wilcox.test(challenger_data[[indicator]], 
                           baseline_data[[indicator]], 
                           paired = TRUE)$p.value
    
    # Calculate Cliff's Delta effect size
    cliff_result <- cliff.delta(challenger_data[[indicator]], 
                                baseline_data[[indicator]])
    
    effect_size <- cliff_result$estimate
    magnitude <- get_magnitude(effect_size)
    
    # Store the results
    results_list[[length(results_list) + 1]] <- data.frame(
      Comparison = paste(challenger_name, "vs", BASELINE_MODEL_NAME),
      Metric = indicator,
      p_value = p_value,
      Cliff_Delta = effect_size,
      Magnitude = magnitude
    )
  }
}

# Combine all results into a single, clean data frame
final_results_df <- do.call(rbind, results_list)

# --- 5. OUTPUT RESULTS ---

# Format the results for better readability
final_results_df$p_value_formatted <- ifelse(final_results_df$p_value < 0.001, 
                                             "< 0.001", 
                                             round(final_results_df$p_value, 3))
final_results_df$Significance <- ifelse(final_results_df$p_value < 0.05, "Significant", "Not Significant")

# Print the final summary table to the console
print("--- Statistical Analysis Summary ---")
print(final_results_df[, c("Comparison", "Metric", "p_value_formatted", "Significance", "Cliff_Delta", "Magnitude")])

# Save the final summary table to a CSV file
output_file_path <- paste0(EXPS_RESULTS_PATH, "statistical_summary.csv")
write.csv(final_results_df, output_file_path, row.names = FALSE)

print(paste("\nStatistical summary saved to:", output_file_path))
