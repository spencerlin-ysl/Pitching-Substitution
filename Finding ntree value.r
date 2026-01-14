library(randomForest)
library(dplyr)
library(ggplot2)
library(Metrics)

# Grid of ntree values to test
ntree_grid <- c(50, 100, 200, 300, 500)

ntree_results <- data.frame(
  ntree = ntree_grid,
  train_rmse = NA,
  test_rmse = NA
)

for (i in seq_along(ntree_grid)) {
  
  nt <- ntree_grid[i]
  
  # Train model
  rf_fit <- randomForest(
    run_value ~ pitch_count + prev_1 + rolling_2_at_bats + 
      rolling_3_at_bats + ff_quality + ff_spin_quality + ff_release_point,
    data = model_data,
    ntree = nt,
    mtry = 1,
    nodesize = 60,
    importance = TRUE
  )
  
  # Predictions
  model_data$pred <- predict(rf_fit, newdata = model_data)
  testing_model_data$pred <- predict(rf_fit, newdata = testing_model_data)
  
  # RMSEs
  ntree_results$train_rmse[i] <- rmse(model_data$run_value, model_data$pred)
  ntree_results$test_rmse[i]  <- rmse(testing_model_data$run_value, testing_model_data$pred)
}

print(ntree_results)

# Plot Testing RMSE vs ntree
ggplot(ntree_results, aes(x = ntree, y = test_rmse)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Testing RMSE vs ntree",
    x = "Number of Trees (ntree)",
    y = "Testing RMSE"
  ) +
  theme_minimal()

