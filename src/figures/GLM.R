library(readr)
library(lme4)
library(lmerTest)
library(effectsize)

setwd("J:/submission_code_revision/src/figures")

# ============================================================================
# Load data from unified evaluation CSV
# ============================================================================
RESULTS_FILE <- "../results/latency/evaluation_fingers.csv"
df_raw <- read_csv(RESULTS_FILE, col_types = cols(condition = col_character()))

# Filter to all-fingers condition and fingers experiment only
df_raw <- df_raw[df_raw$experiment_type == "fingers" & df_raw$condition == "01234", ]

# Keep only correct trials
df_raw <- df_raw[df_raw$correct == 1, ]

# Compute negative latency: nl = max(t_nl1 - t_lift, t_hand - t_lift)
df_raw$nl <- pmax(df_raw$t_nl1 - df_raw$t_lift, df_raw$t_hand - df_raw$t_lift)

# Rename/select columns to match the original analysis
df <- data.frame(
  nl             = df_raw$nl,
  lifted_target  = as.factor(df_raw$lifted_target),
  val_phase      = as.factor(df_raw$block),
  participant    = as.factor(df_raw$participant)
)

cat("Loaded", nrow(df), "trials\n")
cat("Participants:", length(unique(df$participant)), "\n")
cat("Objects:", length(unique(df$lifted_target)), "\n")
cat("Blocks:", length(unique(df$val_phase)), "\n\n")

str(df)
head(df)

# ============================================================================
# Mixed linear model
# ============================================================================
lmm_model <- lmer(nl ~ lifted_target * val_phase + (1|participant),
                  data = df)

summary_model <- summary(lmm_model)

# Extract p-values from fixed effects
fixed_effects <- coef(summary_model)
p_values <- fixed_effects[, "Pr(>|t|)"]

# Apply FDR correction
p_adjusted_fdr <- p.adjust(p_values, method = "fdr")

# Results table
results_table <- data.frame(
  Effect   = rownames(fixed_effects),
  Estimate = fixed_effects[, "Estimate"],
  SE       = fixed_effects[, "Std. Error"],
  t_value  = fixed_effects[, "t value"],
  p_value  = p_values,
  p_FDR    = p_adjusted_fdr
)

print(results_table)

# Generate the ANOVA table for the mixed model
anova_results <- anova(lmm_model)

# Print the standard ANOVA output
print(anova_results)

# Extract the F-statistics and p-values into a custom data frame
f_results_table <- data.frame(
  Effect   = rownames(anova_results),
  F_value  = anova_results$`F value`,
  NumDF    = anova_results$NumDF,
  DenDF    = anova_results$DenDF,
  p_value  = anova_results$`Pr(>F)`
)

# Apply FDR correction to the ANOVA p-values
f_results_table$p_FDR <- p.adjust(f_results_table$p_value, method = "fdr")

print(f_results_table)

