library(randomForest)
library(ModelMetrics)


mtry_grid <- 1:6

results_mtry <- data.frame(
  mtry = mtry_grid,
  train_rmse = NA,
  test_rmse = NA
)

constant_ntree <- 300
constant_nodesize <- 60

for (i in seq_along(mtry_grid)) {
  
  m_val <- mtry_grid[i]
  
  rf_fit <- randomForest(
    run_value ~ pitch_count + prev_1 + rolling_2_at_bats + 
      rolling_3_at_bats + ff_quality + ff_spin_quality + ff_release_point,
    data = model_data,
    ntree = constant_ntree, 
    mtry = m_val, 
    nodesize = constant_nodesize,
    importance = FALSE
  )
  

  model_data$pred <- predict(rf_fit, newdata = model_data)
  testing_model_data$pred <- predict(rf_fit, newdata = testing_model_data)

  results_mtry$train_rmse[i] <- rmse(model_data$run_value, model_data$pred)
  results_mtry$test_rmse[i]  <- rmse(testing_model_data$run_value, testing_model_data$pred)
}

print(results_mtry)

# find the best mtry:
# best_mtry_row <- results_mtry[which.min(results_mtry$test_rmse), ]
# print(best_mtry_row)