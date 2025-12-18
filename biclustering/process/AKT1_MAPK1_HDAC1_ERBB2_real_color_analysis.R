# ============================================================
# Comparative Expression–Mutation Plots for AKKT1, MAPK1, HDAC1, ERBB2
# R
# source("process/AKT1_MAPK1_HDAC1_ERBB2_real_color_analysis.R")
# ============================================================
library(ggplot2)
library(dplyr)

set.seed(123)

# Ensure output directory exists
out_dir <- "../ACGNN/data"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# -----------------------------
# Function to generate mock data
# -----------------------------
simulate_gene_data <- function(gene_name) {
  data.frame(
    Gene = gene_name,
    Expression = rnorm(200, runif(1, 8, 12), 2),
    Mutation_Status = rep(c("WT", "MT"), each = 100),
    Deleterious = sample(c("TRUE", "FALSE", "NA"), 200, replace = TRUE),
    Variant_Classification = sample(c(
      "frameshift_variant", "inframe_deletion", "inframe_insertion",
      "missense_variant", "missense_variant&splice", "splice_acceptor_variant",
      "splice_donor_variant", "splice_region_variant", "stop_gained",
      "stop_gained&frameshift", "stop_gained&protein", "stop_gained&splice_r",
      "WT"
    ), 200, replace = TRUE)
  )
}

# Combine simulated datasets for all four genes
df_all <- bind_rows(
  simulate_gene_data("AKT1"),
  simulate_gene_data("MAPK1"),
  simulate_gene_data("HDAC1"),
  simulate_gene_data("ERBB2")
)

# -----------------------------
# Color palettes
# -----------------------------
deleterious_colors <- c("TRUE" = "#E41A1C", "FALSE" = "#377EB8", "NA" = "#999999")
variant_colors <- c(
  "frameshift_variant" = "#1B9E77",
  "inframe_deletion" = "#D95F02",
  "inframe_insertion" = "#7570B3",
  "missense_variant" = "#E7298A",
  "missense_variant&splice" = "#66A61E",
  "splice_acceptor_variant" = "#E6AB02",
  "splice_donor_variant" = "#A6761D",
  "splice_region_variant" = "#666666",
  "stop_gained" = "#A6CEE3",
  "stop_gained&frameshift" = "#1F78B4",
  "stop_gained&protein" = "#B2DF8A",
  "stop_gained&splice_r" = "#FB9A99",
  "WT" = "#F0F0F0"
)

# -----------------------------
# Function to plot and save figures
# -----------------------------
plot_gene_panels <- function(df, gene_name, out_dir) {
  df_gene <- df %>% filter(Gene == gene_name)
  
  # Panel C: WT vs MT, colored by Deleterious
  pC <- ggplot(df_gene %>% filter(Mutation_Status %in% c("WT", "MT")),
               aes(x = Mutation_Status, y = Expression, fill = Deleterious)) +
    geom_violin(trim = FALSE, alpha = 0.5) +
    geom_boxplot(width = 0.15, outlier.size = 0.5,
                 position = position_dodge(width = 0.9)) +
    scale_fill_manual(values = deleterious_colors) +
    # labs(title = paste0(gene_name, " Expression by Mutation Status"),
    labs(title = gene_name,
         x = "", y = "Expression (log2 values)") +
    theme_bw(base_size = 18) +
    theme(legend.title = element_text(size = 12),
          legend.text = element_text(size = 10),
          plot.title = element_text(hjust = 0.5, face = "bold"))
  
  # Panel D: Expression by Variant Classification
  pD <- ggplot(df_gene, aes(x = Variant_Classification, y = Expression,
                            fill = Variant_Classification)) +
    geom_violin(trim = FALSE, alpha = 0.5) +
    geom_boxplot(width = 0.15, outlier.size = 0.5,
                 position = position_dodge(width = 0.9)) +
    scale_fill_manual(values = variant_colors) +
    # labs(title = paste0(gene_name, " Expression by Variant Classification"),
    labs(title = gene_name,
         x = NULL, y = "Expression (log2 values)") +
    theme_bw(base_size = 18) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 14),
          legend.position = "none",
          plot.title = element_text(hjust = 0.5, face = "bold"))
  
  # Display plots
  print(pC)
  print(pD)
  
  # -----------------------------
  # Save to "data/" directory
  # -----------------------------
  ggsave(file.path(out_dir, paste0("PanelC_", gene_name, ".png")), pC, width = 6, height = 5, dpi = 300)
  ggsave(file.path(out_dir, paste0("PanelD_", gene_name, ".png")), pD, width = 8, height = 5, dpi = 300)
  ggsave(file.path(out_dir, paste0("PanelC_", gene_name, ".pdf")), pC, width = 6, height = 5)
  ggsave(file.path(out_dir, paste0("PanelD_", gene_name, ".pdf")), pD, width = 8, height = 5)
  
  message(paste("Plots saved for", gene_name, "in", out_dir))
}

# -----------------------------
# Run for all four genes
# -----------------------------
genes <- c("AKT1", "MAPK1", "HDAC1", "ERBB2")
for (g in genes) {
  plot_gene_panels(df_all, g, out_dir)
}
