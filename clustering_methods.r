library(tidyverse)
library(cluster)
library(factoextra)
library(kernlab)

# Prepare clustering data
clustering_data <- model_data %>%
  mutate(
    # Convert base runners to binary indicators
    on_1b = ifelse(is.na(on_1b), 0, 1),
    on_2b = ifelse(is.na(on_2b), 0, 1),
    on_3b = ifelse(is.na(on_3b), 0, 1),
    # Convert scored to binary
    scored_binary = ifelse(scored > 0, 1, 0)
  ) %>%
  # Select numeric features for clustering
  select(
    run_value, inning, at_bat_number, pitch_count, 
    at_bat_total_score, rolling_2_at_bats, rolling_3_at_bats,
    mean_ff, mean_ff_spin, mean_release_point,
    on_1b, on_2b, on_3b, outs_when_up, at_bat_order,
    prev_1, ff_quality, ff_spin_quality, ff_release_point,
    scored_binary
  ) %>%
  drop_na()

clustering_data <- clustering_data %>% 
  filter(ff_spin_quality >= -500 & ff_spin_quality <= 500)

# Scale the features
clustering_scaled <- clustering_data %>%
  scale() %>%
  as.data.frame()

# ============================================================
# 1. K-MEANS CLUSTERING
# ============================================================

# Determine optimal k using elbow method
set.seed(123)
fviz_nbclust(clustering_scaled, kmeans, method = "wss", k.max = 10) +
  labs(title = "Elbow Method for Optimal k")

# Determine optimal k using silhouette method
fviz_nbclust(clustering_scaled, kmeans, method = "silhouette", k.max = 10) +
  labs(title = "Silhouette Method for Optimal k")

# Perform K-means with chosen k (e.g., k=4)
kmeans_result <- kmeans(clustering_scaled, centers = 3, nstart = 25, iter.max = 100)

# Add cluster assignments back to original data
clustering_data$kmeans_cluster <- kmeans_result$cluster

# Visualize K-means clusters
fviz_cluster(kmeans_result, data = clustering_scaled,
             geom = "point",
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal(),
             main = "K-means Clustering")

# Summary statistics by cluster
clustering_data %>%
  group_by(kmeans_cluster) %>%
  summarise(
    n = n(),
    avg_run_value = mean(run_value),
    avg_pitch_count = mean(pitch_count),
    avg_ff = mean(mean_ff),
    pct_scored = mean(scored_binary)
  )

# ============================================================
# 2. HIERARCHICAL CLUSTERING
# ============================================================

# For large datasets, sample for hierarchical clustering
set.seed(123)
sample_size <- min(5000, nrow(clustering_scaled))
sample_idx <- sample(1:nrow(clustering_scaled), sample_size)
clustering_sample <- clustering_scaled[sample_idx, ]
clustering_sample_unscaled <- clustering_data[sample_idx, ]

# Calculate distance matrix
dist_matrix <- dist(clustering_sample, method = "euclidean")

# Perform hierarchical clustering
hc_complete <- hclust(dist_matrix, method = "complete")
hc_average <- hclust(dist_matrix, method = "average")
hc_ward <- hclust(dist_matrix, method = "ward.D2")

# Plot dendrograms
plot(hc_ward, main = "Hierarchical Clustering Dendrogram (Ward's Method)",
     xlab = "", sub = "", cex = 0.6)
rect.hclust(hc_ward, k = 3, border = 2:5)

# Cut tree to get cluster assignments
hc_clusters <- cutree(hc_ward, k = 3)

# Visualize hierarchical clusters
fviz_cluster(list(data = clustering_sample, cluster = hc_clusters),
             geom = "point",
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal(),
             main = "Hierarchical Clustering (Ward's Method)")

cluster_summary <- cbind(
  data.frame(cluster = hc_clusters),
  clustering_sample_unscaled
) %>%
  group_by(cluster) %>%
  summarise(across(everything(), mean))

cluster_summary %>% 
  group_by(cluster) %>%
  summarise(
    avg_run_value = mean(run_value),
    avg_pitch_count = mean(pitch_count),
    avg_ff = mean(mean_ff),
    pct_scored = mean(scored_binary)
  )

# ============================================================
# 3. SPECTRAL CLUSTERING
# ============================================================

# For spectral clustering, also use a sample for computational efficiency
set.seed(123)
spectral_sample_size <- min(2000, nrow(clustering_scaled))
spectral_idx <- sample(1:nrow(clustering_scaled), spectral_sample_size)
spectral_sample <- clustering_scaled[spectral_idx, ]
spectral_sample_unscaled <- clustering_data[spectral_idx, ]

# Perform spectral clustering
spec_result <- specc(as.matrix(spectral_sample), centers = 3)

# Add cluster assignments
spectral_clusters <- spec_result@.Data
spectral_sample_unscaled$cluster <- spectral_clusters

# Visualize spectral clusters (using PCA for 2D visualization)
pca_result <- prcomp(spectral_sample)
spectral_viz <- data.frame(
  PC1 = pca_result$x[, 1],
  PC2 = pca_result$x[, 2],
  Cluster = as.factor(spectral_clusters)
)

ggplot(spectral_viz, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  labs(title = "Spectral Clustering Results",
       x = "First Principal Component",
       y = "Second Principal Component") +
  scale_color_brewer(palette = "Set1")

spectral_summary <- spectral_sample_unscaled %>%
  group_by(cluster) %>%
  summarise(
    n = n(),
    avg_run_value = mean(run_value),
    avg_pitch_count = mean(pitch_count),
    avg_ff = mean(mean_ff),
    pct_scored = mean(scored_binary)
  )
spectral_summary

# ============================================================
# COMPARE CLUSTERING METHODS
# ============================================================

# For the sample used in hierarchical clustering, compare all three methods
kmeans_sample <- kmeans(clustering_sample, centers = 3, nstart = 25)

comparison_df <- data.frame(
  kmeans = kmeans_sample$cluster,
  hierarchical = hc_clusters,
  row_id = sample_idx
)

# For spectral, match the indices
spectral_comparison <- data.frame(
  spectral = spectral_clusters,
  row_id = spectral_idx
)

# Calculate adjusted Rand index between methods (on overlapping samples)
library(mclust)
overlapping_idx <- intersect(sample_idx, spectral_idx)
if (length(overlapping_idx) > 0) {
  overlap_pos_sample <- which(sample_idx %in% overlapping_idx)
  overlap_pos_spectral <- which(spectral_idx %in% overlapping_idx)
  
  cat("Adjusted Rand Index (K-means vs Hierarchical):",
      adjustedRandIndex(kmeans_sample$cluster, hc_clusters), "\n")
  cat("Adjusted Rand Index (K-means vs Spectral):",
      adjustedRandIndex(kmeans_sample$cluster[overlap_pos_sample],
                        spectral_clusters[overlap_pos_spectral]), "\n")
}

# ============================================================
# SAVE RESULTS
# ============================================================

# Add all cluster assignments back to model_data
model_data <- model_data %>% 
  filter(ff_spin_quality >= -500 & ff_spin_quality <= 500)

model_data_clustered <- model_data %>%
  mutate(
    on_1b = ifelse(is.na(on_1b), 0, 1),
    on_2b = ifelse(is.na(on_2b), 0, 1),
    on_3b = ifelse(is.na(on_3b), 0, 1),
    scored_binary = ifelse(scored > 0, 1, 0),
    kmeans_cluster = kmeans_result$cluster
  )

# Save clustered data
# write_csv(model_data_clustered, "model_data_with_clusters.csv")

