library(tidyverse)
library(randomForest)
library(glmnet)
library(xgboost)
library(LiblineaR)
library(e1071)
library(FNN)
library(caret)
library(MASS)

# ============================================================
# BOX-COX TRANSFORMATION FOR RESPONSE VARIABLE
# ============================================================

# Box-Cox requires positive values, so shift if needed
y_shift <- min(model_data$run_value) - 0.01
if (y_shift < 0) {
  model_data$run_value_shifted <- model_data$run_value - y_shift
  testing_model_data$run_value_shifted <- testing_model_data$run_value - y_shift
  cat("Shifted run_value by", abs(y_shift), "to make all values positive\n")
} else {
  model_data$run_value_shifted <- model_data$run_value
}

# Find optimal lambda using a sample (for speed)
set.seed(123)
sample_bc <- sample(1:nrow(model_data), min(5000, nrow(model_data)))
lm_temp <- lm(run_value_shifted ~ pitch_count + prev_1 + rolling_2_at_bats + 
                rolling_3_at_bats + ff_quality + ff_spin_quality + ff_release_point,
              data = model_data[sample_bc, ])

bc_result <- boxcox(lm_temp, plotit = TRUE)
lambda_optimal <- bc_result$x[which.max(bc_result$y)]
cat("Optimal Box-Cox lambda:", lambda_optimal, "\n")

# Apply Box-Cox transformation
if (abs(lambda_optimal) < 0.01) {
  model_data$run_value_transformed <- log(model_data$run_value_shifted)
  cat("Using log transformation (lambda ≈ 0)\n")
} else {
  model_data$run_value_transformed <- (model_data$run_value_shifted^lambda_optimal - 1) / lambda_optimal
  testing_model_data$run_value_transformed <- (testing_model_data$run_value_shifted^lambda_optimal - 1) / lambda_optimal
  cat("Using Box-Cox transformation with lambda =", lambda_optimal, "\n")
}

use_transformed <- TRUE  # Set to FALSE to use original run_value

if (use_transformed) {
  cat("\n*** USING TRANSFORMED RESPONSE VARIABLE ***\n")
  model_data$run_value <- model_data$run_value_transformed
}


# ============================================================
# PREPARE DATA
# ============================================================

# Split data into training and testing sets
set.seed(123)

train_data <- model_data
test_data <- testing_model_data

# Define predictor variables
predictors <- c("pitch_count", "prev_1", "rolling_2_at_bats", 
                "rolling_3_at_bats", "ff_quality", "ff_spin_quality", 
                "ff_release_point", "xwoba")

# Prepare matrices for models that need them
X_train <- as.matrix(train_data[, predictors])
y_train <- train_data$run_value_transformed
X_test <- as.matrix(test_data[, predictors])
y_test <- test_data$run_value_transformed

X_train_scaled <- scale(X_train)
X_test_scaled <- scale(X_test, 
                       center = attr(X_train_scaled, "scaled:center"),
                       scale = attr(X_train_scaled, "scaled:scale"))

# ============================================================
# MODEL 1: LASSO REGRESSION
# ============================================================
cat("\n=== Model 1: Penalized Linear Regression (Lasso) ===\n")

# Cross-validation to find optimal lambda
set.seed(123)
lasso_cv <- cv.glmnet(X_train, y_train, alpha = 1, nfolds = 10)

# Fit lasso regression with optimal lambda
lasso_fit <- glmnet(X_train, y_train, alpha = 1, lambda = lasso_cv$lambda.min)

# Predictions
lasso_pred_train <- predict(lasso_fit, newx = X_train, s = lasso_cv$lambda.min)
lasso_pred_test <- predict(lasso_fit, newx = X_test, s = lasso_cv$lambda.min)

# Plot lambda vs MSE
plot(lasso_cv, main = "Lasso Regression: Cross-Validation")
cat("Optimal lambda for Lasso:", lasso_cv$lambda.min, "\n")

# Coefficients (note which are shrunk to zero)
lasso_coefs <- coef(lasso_fit, s = lasso_cv$lambda.min)
cat("Lasso Regression Coefficients:\n")
print(lasso_coefs)

# ============================================================
# MODEL 2: KNN REGRESSION
# ============================================================
cat("\n=== Model 2: K-Nearest Neighbors Regression ===\n")

# Find optimal k using cross-validation
k_values <- seq(150, 300, by = 25)
knn_cv_errors <- numeric(length(k_values))

set.seed(123)
for (i in seq_along(k_values)) {
  # 5-fold cross-validation
  folds <- createFolds(y_train, k = 5)
  fold_errors <- numeric(5)
  
  for (j in seq_along(folds)) {
    train_fold <- X_train_scaled[-folds[[j]], ]
    test_fold <- X_train_scaled[folds[[j]], ]
    train_y <- y_train[-folds[[j]]]
    test_y <- y_train[folds[[j]]]
    
    pred <- knn.reg(train_fold, test_fold, train_y, k = k_values[i])$pred
    fold_errors[j] <- mean((test_y - pred)^2)
  }
  knn_cv_errors[i] <- mean(fold_errors)
}

# Find optimal k
optimal_k <- k_values[which.min(knn_cv_errors)]
cat("Optimal k:", optimal_k, "\n")
optimal_k <- 200

# Plot k vs CV error
plot(k_values, sqrt(knn_cv_errors), type = "b", 
     xlab = "k", ylab = "CV RMSE",
     main = "KNN: Cross-Validation for Optimal k")
abline(v = optimal_k, col = "red", lty = 2)

# Fit final KNN model
knn_pred_train <- knn.reg(X_train_scaled, X_train_scaled, y_train, k = optimal_k)$pred
knn_pred_test <- knn.reg(X_train_scaled, X_test_scaled, y_train, k = optimal_k)$pred

# ============================================================
# MODEL 3: SVM REGRESSION
# ============================================================
cat("\n=== Model 3: Support Vector Machine Regression ===\n")

# Tune SVM parameters
set.seed(123)
svm_tune <- tune(svm, train.x = X_train_scaled, train.y = y_train,
                 ranges = list(
                   epsilon = c(0.1, 0.2),
                   cost = c(10, 100)
                 ),
                 tunecontrol = tune.control(cross = 5))

cat("Best SVM parameters:\n")
print(svm_tune$best.parameters)

# Fit SVM with best parameters
svm_fit <- svm_tune$best.model
svm_ll <- LiblineaR(
  data = X_train_scaled,
  target = y_train,
  type = 11,    # L2-regularized SVR
  cost = 10,
  epsilon = 0.1
)

# Predictions
svm_pred_train <- predict(svm_ll, X_train_scaled)
svm_pred_train <- svm_pred_train$predictions
svm_pred_test <- predict(svm_ll, X_test_scaled)
svm_pred_test <- svm_pred_test$predictions

# Variable Importance for Linear SVM
# Based on absolute value of coefficients
svm_coefs <- svm_ll$W
svm_importance <- data.frame(
  Variable = colnames(svm_coefs),
  Coefficient = as.vector(svm_coefs),
  Importance = abs(as.vector(svm_coefs))
) %>%
  arrange(desc(Importance))

cat("\nSVM Variable Importance (based on absolute coefficients):\n")
print(svm_importance)

# Plot variable importance
ggplot(svm_importance, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  geom_text(aes(label = round(Importance, 3)), hjust = -0.1, size = 3) +
  coord_flip() +
  labs(title = "SVM: Variable Importance",
       subtitle = "Based on absolute value of coefficients",
       x = "Variable", y = "Absolute Coefficient Value") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

# Also show coefficients with sign
ggplot(svm_importance, aes(x = reorder(Variable, Coefficient), y = Coefficient)) +
  geom_col(aes(fill = Coefficient > 0), alpha = 0.8) +
  geom_text(aes(label = round(Coefficient, 3)), 
            hjust = ifelse(svm_importance$Coefficient > 0, -0.1, 1.1), 
            size = 3) +
  coord_flip() +
  labs(title = "SVM: Coefficient Values",
       subtitle = "Positive = increases run value, Negative = decreases run value",
       x = "Variable", y = "Coefficient") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "none") +
  scale_fill_manual(values = c("FALSE" = "coral", "TRUE" = "steelblue"))


# ============================================================
# MODEL 4: RANDOM FOREST
# ============================================================
cat("\n=== Model 4: Random Forest ===\n")

set.seed(123)
rf_fit <- randomForest(
  run_value ~ pitch_count + prev_1 + rolling_2_at_bats + 
    rolling_3_at_bats + ff_quality + ff_spin_quality + ff_release_point,
  data = train_data,
  ntree = 300,
  mtry = 3,  # sqrt(7) ≈ 3 for regression
  nodesize = 60,
  importance = TRUE
)

# Predictions
rf_pred_train <- predict(rf_fit, train_data)
rf_pred_test <- predict(rf_fit, test_data)

# Variable importance
cat("Random Forest Variable Importance:\n")
print(importance(rf_fit))
varImpPlot(rf_fit, main = "Random Forest: Variable Importance")

# ============================================================
# MODEL 5: GRADIENT BOOSTING (XGBoost)
# ============================================================
cat("\n=== Model 5: Gradient Boosting (XGBoost) ===\n")

# Prepare data for XGBoost
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test, label = y_test)

# Set parameters
xgb_params <- list(
  objective = "reg:squarederror",
  eta = 0.01,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  eval_metric = "rmse"
)

# Cross-validation to find optimal nrounds
set.seed(123)
xgb_cv <- xgb.cv(
  params = xgb_params,
  data = dtrain,
  nrounds = 500,
  nfold = 10,
  early_stopping_rounds = 50,
  verbose = 0
)

# Optimal number of rounds
best_nrounds <- xgb_cv$best_iteration
cat("Optimal number of rounds for XGBoost:", best_nrounds, "\n")

# Train final model
set.seed(123)
xgb_fit <- xgboost(
  params = xgb_params,
  data = dtrain,
  nrounds = best_nrounds,
  verbose = 0
)

# Predictions
xgb_pred_train <- predict(xgb_fit, dtrain)
xgb_pred_test <- predict(xgb_fit, dtest)

# xgb_pred_train <- predict(xgb_fit_robust, dtrain)
# xgb_pred_test <- predict(xgb_fit_robust, dtest)

# Variable importance
xgb_importance <- xgb.importance(model = xgb_fit, 
                                 feature_names = colnames(X_train))

cat("XGBoost Variable Importance:\n")
print(xgb_importance)
xgb.plot.importance(xgb_importance, main = "XGBoost: Variable Importance")

# ============================================================
# EVALUATION METRICS FUNCTION
# ============================================================

calculate_metrics <- function(actual, predicted) {
  
  actual <- as.vector(actual)
  predicted <- as.vector(predicted)
  
  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  r_squared <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
  
  return(data.frame(
    RMSE = rmse,
    MAE = mae,
    R_squared = r_squared
  ))
}

# ============================================================
# COMPARE ALL MODELS
# ============================================================

cat("\n=== MODEL COMPARISON ===\n")

# Calculate metrics for all models
results <- data.frame(
  Model = c("Penalized Linear (Lasso)", "KNN", "SVM", 
            "Random Forest", "XGBoost"),
  Train_RMSE = c(
    calculate_metrics(y_train, lasso_pred_train)$RMSE,
    calculate_metrics(y_train, knn_pred_train)$RMSE,
    calculate_metrics(y_train, svm_pred_train)$RMSE,
    calculate_metrics(y_train, rf_pred_train)$RMSE,
    calculate_metrics(y_train, xgb_pred_train)$RMSE
  ),
  Test_RMSE = c(
    calculate_metrics(y_test, lasso_pred_test)$RMSE,
    calculate_metrics(y_test, knn_pred_test)$RMSE,
    calculate_metrics(y_test, svm_pred_test)$RMSE,
    calculate_metrics(y_test, rf_pred_test)$RMSE,
    calculate_metrics(y_test, xgb_pred_test)$RMSE
  ),
  Train_R2 = c(
    calculate_metrics(y_train, lasso_pred_train)$R_squared,
    calculate_metrics(y_train, knn_pred_train)$R_squared,
    abs(calculate_metrics(y_train, svm_pred_train)$R_squared),
    calculate_metrics(y_train, rf_pred_train)$R_squared,
    calculate_metrics(y_train, xgb_pred_train)$R_squared
  ),
  Test_R2 = c(
    calculate_metrics(y_test, lasso_pred_test)$R_squared,
    calculate_metrics(y_test, knn_pred_test)$R_squared,
    abs(calculate_metrics(y_test, svm_pred_test)$R_squared),
    calculate_metrics(y_test, rf_pred_test)$R_squared,
    calculate_metrics(y_test, xgb_pred_test)$R_squared
  )
)

# Round for display
results_rounded <- results %>%
  mutate(across(where(is.numeric), ~round(., 4)))

print(results_rounded)

# ============================================================
# VISUALIZE MODEL COMPARISON
# ============================================================

# Plot 1: Test RMSE Comparison
results_long <- results %>%
  select(Model, Test_RMSE) %>%
  arrange(Test_RMSE)

ggplot(results_long, aes(x = reorder(Model, -Test_RMSE), y = Test_RMSE, fill = Model)) +
  geom_col(alpha = 0.8) +
  geom_text(aes(label = round(Test_RMSE, 4)), vjust = -0.5, size = 3.5) +
  labs(title = "Model Comparison: Test RMSE",
       subtitle = "Lower is better",
       x = "Model", y = "Root Mean Squared Error") +
  theme_minimal() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "bold")) +
  scale_fill_brewer(palette = "Set2")

# Plot 2: R-squared Comparison
results_r2 <- results %>%
  select(Model, Test_R2) %>%
  arrange(desc(Test_R2))

ggplot(results_r2, aes(x = reorder(Model, Test_R2), y = Test_R2, fill = Model)) +
  geom_col(alpha = 0.8) +
  geom_text(aes(label = round(Test_R2, 4)), vjust = -0.5, size = 3.5) +
  labs(title = "Model Comparison: Test R²",
       subtitle = "Higher is better",
       x = "Model", y = "R-squared") +
  theme_minimal() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "bold")) +
  scale_fill_brewer(palette = "Set2") +
  ylim(0, max(results_r2$Test_R2) * 1.1)

# Plot 3: Train vs Test Performance (checking for overfitting)
overfit_check <- results %>%
  select(Model, Train_RMSE, Test_RMSE) %>%
  pivot_longer(cols = c(Train_RMSE, Test_RMSE),
               names_to = "Dataset",
               values_to = "RMSE") %>%
  mutate(Dataset = gsub("_RMSE", "", Dataset))

ggplot(overfit_check, aes(x = Model, y = RMSE, fill = Dataset)) +
  geom_col(position = "dodge", alpha = 0.8) +
  labs(title = "Train vs Test RMSE: Overfitting Check",
       subtitle = "Large gap indicates overfitting",
       x = "Model", y = "RMSE") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "bold")) +
  scale_fill_manual(values = c("Train" = "lightblue", "Test" = "coral"))

# Plot 4: All metrics side by side
results_all_metrics <- results %>%
  select(Model, Test_RMSE, Test_MAE, Test_R2) %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value") %>%
  mutate(Metric = gsub("Test_", "", Metric))

ggplot(results_all_metrics, aes(x = Model, y = Value, fill = Metric)) +
  geom_col(position = "dodge", alpha = 0.8) +
  facet_wrap(~Metric, scales = "free_y") +
  labs(title = "Comprehensive Model Comparison",
       x = "Model", y = "Metric Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "bold"),
        legend.position = "none") +
  scale_fill_brewer(palette = "Set1")

# ============================================================
# RESIDUAL PLOTS FOR BEST MODEL
# ============================================================

# Find best model based on Test RMSE
best_model_idx <- which.min(results$Test_RMSE)
best_model_name <- results$Model[best_model_idx]

# Get predictions from best model
best_pred_test <- switch(best_model_name,
                         "Penalized Linear (Lasso)" = as.vector(lasso_pred_test),
                         "KNN" = knn_pred_test,
                         "SVM" = svm_pred_test,
                         "Random Forest" = rf_pred_test,
                         "XGBoost" = xgb_pred_test)

cat("\nBest model:", best_model_name, "\n")

residuals_df <- data.frame(
  Predicted = best_pred_test,
  Actual = y_test,
  Residual = y_test - best_pred_test
)

# Residual vs Predicted
p1 <- ggplot(residuals_df, aes(x = Predicted, y = Residual)) +
  geom_point(alpha = 0.4, color = "steelblue") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  geom_smooth(method = "loess", color = "darkred", se = TRUE) +
  labs(title = paste(best_model_name, ": Residual Plot"),
       x = "Predicted Run Value", y = "Residuals") +
  theme_minimal()

# Actual vs Predicted
p2 <- ggplot(residuals_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.4, color = "darkgreen") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = paste(best_model_name, ": Actual vs Predicted"),
       x = "Actual Run Value", y = "Predicted Run Value") +
  theme_minimal()

# Q-Q plot for normality of residuals
p3 <- ggplot(residuals_df, aes(sample = Residual)) +
  stat_qq(color = "steelblue", alpha = 0.6) +
  stat_qq_line(color = "red", linetype = "dashed") +
  labs(title = paste(best_model_name, ": Q-Q Plot"),
       x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_minimal()

# Histogram of residuals
p4 <- ggplot(residuals_df, aes(x = Residual)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, 
                 fill = "steelblue", alpha = 0.7, color = "white") +
  geom_density(color = "darkred", linewidth = 1) +
  labs(title = paste(best_model_name, ": Residual Distribution"),
       x = "Residuals", y = "Density") +
  theme_minimal()

gridExtra::grid.arrange(p1, p2, p3, p4, ncol = 2)

# ============================================================
# SAVE RESULTS
# ============================================================

# Save comparison table
# write.csv(results_rounded, "model_comparison_results.csv", row.names = FALSE)

# Save predictions for further analysis
# predictions_df <- data.frame(
#   actual = y_test,
#   lasso = as.vector(lasso_pred_test),
#   knn = knn_pred_test,
#   svm = svm_pred_test,
#   rf = rf_pred_test,
#   xgboost = xgb_pred_test
# )
# write.csv(predictions_df, "test_predictions.csv", row.names = FALSE)