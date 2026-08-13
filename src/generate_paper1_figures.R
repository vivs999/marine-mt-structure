#!/usr/bin/env Rscript
# Paper 1 publication figures - ggplot2 / R
# Outputs to results/figures/paper1/

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(viridis)
  library(RColorBrewer)
  library(cowplot)
  library(scales)
  library(reshape2)
  library(dplyr)
  library(readr)
})

# Run from the repository root.
ROOT      <- "."
RESULTS   <- file.path(ROOT, "results", "models", "multitarget")
OUTDIR    <- file.path(ROOT, "results", "figures", "paper1")
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

# Theme
theme_paper <- function() {
  theme_cowplot(font_size = 11) +
    theme(
      plot.title       = element_text(face = "bold", size = 12),
      plot.subtitle    = element_text(size = 9, color = "grey40"),
      axis.title       = element_text(size = 10),
      axis.text        = element_text(size = 9),
      legend.text      = element_text(size = 9),
      legend.title     = element_text(size = 9, face = "bold"),
      strip.text       = element_text(face = "bold", size = 10),
      panel.grid.major = element_line(color = "grey92", linewidth = 0.4),
      panel.grid.minor = element_blank()
    )
}

HABITAT_COLORS <- c(
  "coastal_estuarine" = "#1565C0",
  "open_ocean"        = "#2E7D32",
  "polar"             = "#6A1B9A",
  "hydrothermal_vent" = "#BF360C"
)

save_fig <- function(p, name, width = 7, height = 5) {
  path <- file.path(OUTDIR, name)
  ggsave(path, plot = p, width = width, height = height, dpi = 300, bg = "white")
  cat(sprintf("  saved -> results/figures/paper1/%s\n", name))
}

load_summary <- function(dir) {
  path <- file.path(RESULTS, dir, "summary_table.csv")
  if (!file.exists(path)) return(NULL)
  read_csv(path, show_col_types = FALSE)
}

# Figure 2: Zn geometry validation
cat("[2/6] Zn geometry validation\n")

mb <- read_csv(file.path(ROOT, "data", "processed", "metalbound_features.csv"),
               show_col_types = FALSE) |>
  filter(!is.na(mb_mean_zn_s_dist), !is.na(mb_mean_szns_angle))

p_dist <- ggplot(mb, aes(x = "Boltz-predicted", y = mb_mean_zn_s_dist)) +
  geom_violin(fill = "#1565C0", alpha = 0.65, color = NA) +
  geom_boxplot(width = 0.08, outlier.shape = NA, color = "white", linewidth = 0.8) +
  geom_hline(yintercept = 2.33, linetype = "dashed", color = "#C62828", linewidth = 0.9) +
  annotate("text", x = 1, y = 2.335, label = "Crystal avg: 2.33 Å",
           color = "#C62828", size = 3.1, hjust = 0.5, vjust = -0.4) +
  annotate("text", x = 1, y = min(mb$mb_mean_zn_s_dist, na.rm=TRUE) + 0.01,
           label = sprintf("Mean = %.3f Å\nn = %d", mean(mb$mb_mean_zn_s_dist, na.rm=TRUE), nrow(mb)),
           size = 3, hjust = 0.5, vjust = 0, color = "grey30") +
  labs(title = "Zn-S Bond Length", x = NULL, y = "Zn-S distance (Å)") +
  theme_paper() +
  theme(axis.ticks.x = element_blank())

p_angle <- ggplot(mb, aes(x = "Boltz-predicted", y = mb_mean_szns_angle)) +
  geom_violin(fill = "#2E7D32", alpha = 0.65, color = NA) +
  geom_boxplot(width = 0.08, outlier.shape = NA, color = "white", linewidth = 0.8) +
  geom_hline(yintercept = 109.5, linetype = "dashed", color = "#C62828", linewidth = 0.9) +
  annotate("text", x = 1, y = 110.3, label = "Ideal tetrahedral: 109.5°",
           color = "#C62828", size = 3.1, hjust = 0.5, vjust = -0.4) +
  annotate("text", x = 1, y = min(mb$mb_mean_szns_angle, na.rm=TRUE) + 0.3,
           label = sprintf("Mean = %.1f°\nn = %d", mean(mb$mb_mean_szns_angle, na.rm=TRUE), nrow(mb)),
           size = 3, hjust = 0.5, vjust = 0, color = "grey30") +
  labs(title = "S-Zn-S Angle", x = NULL, y = "S-Zn-S angle (°)") +
  theme_paper() +
  theme(axis.ticks.x = element_blank())

fig2 <- (p_dist | p_angle) +
  plot_annotation(
    title = "Figure 2 - Zn2+ Coordination Geometry: Boltz Predictions vs Crystal Reference",
    theme = theme(plot.title = element_text(face = "bold", size = 11))
  )
save_fig(fig2, "fig2_zn_geometry_validation.pdf", width = 8, height = 4.5)
save_fig(fig2, "fig2_zn_geometry_validation.png", width = 8, height = 4.5)

# Figure 3: Ablation bar chart
cat("[3/6] Ablation bar chart\n")

s_struct <- load_summary("structural")
s_esm2   <- load_summary("esm2")
s_mb     <- load_summary("metalbound+esm2")

get_cv_acc <- function(df) {
  if (is.null(df)) return(NA)
  df |> filter(target == "habitat_type", model == "random_forest") |>
    pull(cv_accuracy_mean) |> first()
}
get_cv_sd <- function(df) {
  if (is.null(df)) return(NA)
  df |> filter(target == "habitat_type", model == "random_forest") |>
    pull(cv_accuracy_std) |> first()
}

# Ablation reads the repeated stratified group CV results, not the single
# hold-out.  See src/robust_eval.py and the README.
read_robust <- function(name) {
  p <- file.path(ROOT, "results", "robust_eval", paste0(name, ".json"))
  if (!file.exists(p)) return(NULL)
  jsonlite::fromJSON(p)
}
r_struct <- read_robust("structural")
r_esm2   <- read_robust("esm2")
r_mb     <- read_robust("metalbound_esm2")

ablation <- tibble(
  feature_set = factor(
    c("Structural\n(apo ESMFold)", "ESM-2\n(sequence only)", "Metalbound\n+ ESM-2"),
    levels = c("Structural\n(apo ESMFold)", "ESM-2\n(sequence only)", "Metalbound\n+ ESM-2")
  ),
  cv_acc = c(r_struct$habitat_accuracy_mean, r_esm2$habitat_accuracy_mean, r_mb$habitat_accuracy_mean),
  cv_sd  = c(r_struct$habitat_accuracy_sd,   r_esm2$habitat_accuracy_sd,   r_mb$habitat_accuracy_sd)
)

gain_esm2 <- (ablation$cv_acc[2] - ablation$cv_acc[1]) * 100
gain_mb   <- (ablation$cv_acc[3] - ablation$cv_acc[2]) * 100

fig3 <- ggplot(ablation, aes(x = feature_set, y = cv_acc, fill = feature_set)) +
  geom_col(width = 0.55, color = "grey30", linewidth = 0.4) +
  geom_errorbar(aes(ymin = cv_acc - cv_sd, ymax = cv_acc + cv_sd),
                width = 0.18, linewidth = 0.7, color = "grey30") +
  geom_text(aes(label = percent(cv_acc, accuracy = 0.1)),
            vjust = -1.6, size = 3.8, fontface = "bold") +
  annotate("segment", x = 1, xend = 2, y = 0.945, yend = 0.945,
           color = "#2E7D32", linewidth = 0.8,
           arrow = arrow(ends = "last", length = unit(0.15, "cm"))) +
  annotate("text", x = 1.5, y = 0.962,
           label = sprintf("+%.1f points\n(sequence embeddings)", gain_esm2),
           color = "#2E7D32", size = 3.2) +
  annotate("segment", x = 2, xend = 3, y = 0.900, yend = 0.900,
           color = "grey45", linewidth = 0.7, linetype = "22") +
  annotate("text", x = 2.5, y = 0.917,
           label = sprintf("+%.1f points, within noise\n(Zn geometry adds nothing)", gain_mb),
           color = "grey35", size = 3.2) +
  scale_fill_manual(values = c("#90CAF9", "#1565C0", "#1565C0")) +
  scale_y_continuous(labels = percent_format(), expand = c(0, 0)) +
  coord_cartesian(ylim = c(0.6, 1.0)) +
  labs(
    title    = "Figure 3 - Ablation: habitat classification accuracy",
    subtitle = sprintf(
      "10 x 5-fold stratified group CV (n = %d records, %d species). Error bars are 1 SD across repeats.",
      r_mb$n_records, r_mb$n_species),
    x = NULL, y = "Accuracy"
  ) +
  theme_paper() +
  theme(legend.position = "none")

save_fig(fig3, "fig3_ablation_bar.pdf", width = 7, height = 5)
save_fig(fig3, "fig3_ablation_bar.png", width = 7, height = 5)

# Figure 4: Confusion matrix
cat("[4/6] Confusion matrix\n")

preds_path <- file.path(RESULTS, "metalbound+esm2", "habitat_type", "random_forest_predictions.csv")
if (file.exists(preds_path)) {
  preds <- read_csv(preds_path, show_col_types = FALSE)

  labels <- sort(unique(c(preds$y_true, preds$y_pred)))
  cm <- as.data.frame(table(True = factor(preds$y_true, labels), Pred = factor(preds$y_pred, labels)))
  cm_wide <- reshape2::dcast(cm, True ~ Pred, value.var = "Freq", fill = 0)
  mat <- as.matrix(cm_wide[, -1])
  rownames(mat) <- cm_wide$True

  norm_mat <- sweep(mat, 1, rowSums(mat), "/")
  cm_long <- reshape2::melt(norm_mat, varnames = c("True", "Pred"), value.name = "Proportion")
  raw_long <- reshape2::melt(mat, varnames = c("True", "Pred"), value.name = "Count")
  cm_long$Count <- raw_long$Count
  cm_long$label <- sprintf("%.2f\n(n=%d)", cm_long$Proportion, cm_long$Count)

  acc <- sum(diag(mat)) / sum(mat)

  fig4 <- ggplot(cm_long, aes(x = Pred, y = True, fill = Proportion)) +
    geom_tile(color = "white", linewidth = 0.8) +
    geom_text(aes(label = label,
                  color = ifelse(Proportion > 0.6, "white", "black")),
              size = 3.2, lineheight = 1.2) +
    scale_fill_distiller(palette = "Blues", direction = 1, limits = c(0, 1),
                         name = "Proportion") +
    scale_color_identity() +
    scale_x_discrete(labels = function(x) gsub("_", "\n", x)) +
    scale_y_discrete(labels = function(x) gsub("_", "\n", x)) +
    labs(
      title    = "Figure 4 - Habitat Classification Confusion Matrix",
      subtitle = sprintf("RF, metalbound+ESM-2 - test accuracy = %.1f%%  (n=%d)", acc*100, sum(mat)),
      x = "Predicted label", y = "True label"
    ) +
    theme_paper() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))

  save_fig(fig4, "fig4_confusion_matrix.pdf", width = 6.5, height = 5.5)
  save_fig(fig4, "fig4_confusion_matrix.png", width = 6.5, height = 5.5)
} else {
  cat("  SKIP fig4: predictions CSV not found\n")
}

# Figure 5: SST scatter
cat("[5/6] SST scatter\n")

sst_path <- file.path(RESULTS, "metalbound+esm2", "sst_mean_c", "random_forest_predictions.csv")
if (file.exists(sst_path)) {
  sst <- read_csv(sst_path, show_col_types = FALSE)
  r_val <- cor(sst$y_true, sst$y_pred)
  r2    <- 1 - sum((sst$y_true - sst$y_pred)^2) / sum((sst$y_true - mean(sst$y_true))^2)
  lim   <- range(c(sst$y_true, sst$y_pred)) + c(-1.5, 1.5)

  # Join habitat label if uniprot_id is available
  has_uid <- "uniprot_id" %in% names(sst)
  if (has_uid) {
    meta <- read_csv(file.path(ROOT, "data", "processed", "integrated_v2.csv"),
                     show_col_types = FALSE) |>
      select(uniprot_id, habitat_type) |> distinct()
    sst <- left_join(sst, meta, by = "uniprot_id")
  }

  fig5 <- ggplot(sst, aes(x = y_true, y = y_pred)) +
    {if (has_uid && "habitat_type" %in% names(sst))
        geom_point(aes(color = habitat_type), size = 2.2, alpha = 0.8)
     else
        geom_point(color = "#1565C0", size = 2.2, alpha = 0.8)} +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                color = "#C62828", linewidth = 0.9) +
    {if (has_uid && "habitat_type" %in% names(sst))
        scale_color_manual(values = HABITAT_COLORS, name = "Habitat",
                           labels = function(x) gsub("_", " ", tools::toTitleCase(x)))
     else NULL} +
    annotate("text", x = lim[1] + 0.5, y = lim[2] - 0.5,
             label = sprintf("Pearson r = %.3f\nR² = %.3f\nn = %d", r_val, r2, nrow(sst)),
             hjust = 0, vjust = 1, size = 3.5,
             color = "grey20",
             family = "mono") +
    coord_fixed(xlim = lim, ylim = lim) +
    labs(
      title    = "Figure 5 - Sea Surface Temperature Prediction",
      subtitle = "RF, metalbound+ESM-2 (biooracle_sampled rows only)",
      x = "True SST (°C)", y = "Predicted SST (°C)"
    ) +
    theme_paper() +
    theme(legend.position = c(0.72, 0.25))

  save_fig(fig5, "fig5_sst_scatter.pdf", width = 6, height = 5.5)
  save_fig(fig5, "fig5_sst_scatter.png", width = 6, height = 5.5)
} else {
  cat("  SKIP fig5: predictions CSV not found\n")
}

# Figure 6: Results heatmap
cat("[6/6] Results heatmap\n")

feature_dirs <- c(
  "Structural\n(apo)" = "structural",
  "ESM-2\nonly"       = "esm2",
  "Metalbound\n+ESM-2"= "metalbound+esm2"
)

targets <- c("habitat_type", "sst_mean_c", "do_mean_mlL", "ph_mean", "depth_mean_m", "salinity_mean_psu")
target_labels <- c("Habitat type\n(accuracy)", "SST °C\n(Pearson r)",
                   "DO mL/L\n(Pearson r)", "pH\n(Pearson r)",
                   "Depth m\n(Pearson r)", "Salinity psu\n(Pearson r)")

hm_rows <- list()
for (i in seq_along(feature_dirs)) {
  df <- load_summary(feature_dirs[i])
  if (is.null(df)) next
  for (j in seq_along(targets)) {
    tgt <- targets[j]
    row <- df |> filter(target == tgt, model == "random_forest")
    if (nrow(row) == 0) next
    val <- if (tgt == "habitat_type") row$cv_accuracy_mean[1] else row$cv_pearson_mean[1]
    hm_rows[[length(hm_rows)+1]] <- tibble(
      feature = names(feature_dirs)[i],
      target  = target_labels[j],
      value   = val
    )
  }
}
hm <- bind_rows(hm_rows) |>
  mutate(
    feature = factor(feature, levels = names(feature_dirs)),
    target  = factor(target, levels = rev(target_labels)),
    label   = sprintf("%.2f", value)
  )

fig6 <- ggplot(hm, aes(x = feature, y = target, fill = value)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = label,
                color = ifelse(value > 0.65 | value < -0.1, "white", "grey15")),
            size = 3.6, fontface = "bold") +
  scale_fill_distiller(palette = "RdYlGn", direction = 1,
                       limits = c(-0.35, 1.0), oob = squish,
                       name = "CV Accuracy /\nCV Pearson r") +
  scale_color_identity() +
  labs(
    title    = "Figure 6 - Model Performance Across All Targets and Feature Sets",
    subtitle = "Random Forest, 5-fold CV (n = 349 biooracle-sampled rows)",
    x = "Feature set", y = NULL
  ) +
  theme_paper() +
  theme(
    axis.text.x  = element_text(angle = 0),
    axis.ticks.y = element_blank(),
    panel.grid   = element_blank()
  )

save_fig(fig6, "fig6_results_heatmap.pdf", width = 7, height = 5)
save_fig(fig6, "fig6_results_heatmap.png", width = 7, height = 5)

cat("\nDone. All figures -> results/figures/paper1/\n")
