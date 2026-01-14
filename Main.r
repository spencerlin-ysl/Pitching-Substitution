library(baseballr)
library(tidyverse)
library(pROC)

statcast_25 <- read_csv("statcast_2025.csv")
statcast_25 <- statcast_25 %>% map_df(rev) %>% 
  filter(game_type == "R")
weights <- read_csv("weights.csv")

starters <- statcast_25 %>%
  filter(inning == 1) %>%
  group_by(game_date, inning_topbot, home_team, away_team) %>%
  slice(1) %>% 
  transmute(game_date, pitcher = pitcher) %>%
  ungroup()

end_5 <- statcast_25 %>%
  filter(inning == 5) %>%
  group_by(game_date, inning_topbot, home_team, away_team) %>% 
  slice_tail() %>% 
  transmute(game_date, pitcher_end = pitcher) %>%
  ungroup()

qualified <- inner_join(starters, end_5, by = c("game_date", "home_team", "away_team", "inning_topbot")) %>%
  filter(pitcher == pitcher_end)

statcast_25_sp <- statcast_25 %>%
  semi_join(qualified,
            by = c("game_date", "inning_topbot", "home_team", "away_team", "pitcher"))

statcast_25_sp <- statcast_25_sp %>% 
  mutate(single = events == "single",
         double = events == "double",
         triple = events == "triple",
         home_run = events == "home_run",
         walk = events == "walk",
         hit_by_pitch = events == "hit_by_pitch",
         `Swinging Strike` = description == "swinging_strike",
         Ahead_of_count = balls < strikes,
         Behind_count = balls > strikes,
         exit_velo_95_up = launch_speed >= 95,
         exit_velo_80_down = launch_speed <= 80 & events != "foul",
         Out = events %in% c(
           "grounded_into_double_play", "strikeout", "field_out", "force_out",
           "fielders_choice_out", "double_play", "fielders_choice",
           "sac_bunt", "strikeout_double_play", "sac_fly_double_play", "triple_play"
         ), 
         sac_fly = events == "sac_fly",
         Other = !(single | double | triple | home_run | walk | hit_by_pitch |
                     `Swinging Strike` | Out | sac_fly | Ahead_of_count |
                     Behind_count | exit_velo_95_up | exit_velo_80_down))

weights_vec <- setNames(weights$Weight, weights$Event)

statcast_25_sp <- statcast_25_sp[rev(rownames(statcast_25_sp)),]

statcast_25_sp <- statcast_25_sp %>% 
  rowwise() %>%
  mutate(
    PE_per_pitch = sum(
      c_across(all_of(names(weights_vec))) * weights_vec,
      na.rm = TRUE
    )
  ) %>%
  ungroup() %>% 
  group_by(game_date, pitcher) %>% 
  mutate(`Pitch Effectiveness` = cumsum(PE_per_pitch))

statcast_25_sp <- statcast_25_sp %>% 
  group_by(game_date, pitcher) %>%
  mutate(scored = post_bat_score - lag(post_bat_score), 
         inning_change = inning != lag(inning)) %>%
  ungroup()

statcast_25_sp <- statcast_25_sp %>%
  group_by(game_date, pitcher, at_bat_number) %>%
  mutate(
    # Get the first and last pitch effectiveness scores for each at-bat
    first_pitch_score = first(`Pitch Effectiveness`),
    last_pitch_score = last(`Pitch Effectiveness`),
    # Calculate the difference (last - first)
    at_bat_total_score = last_pitch_score - first_pitch_score
  ) %>%
  ungroup() %>%
  # Create a dataset with one row per at-bat for rolling calculations
  group_by(game_date, pitcher, at_bat_number) %>%
  slice(1) %>%
  ungroup() %>%
  # IMPORTANT: Arrange by pitcher, game_date, and at_bat_number to ensure correct order
  arrange(pitcher, game_date, at_bat_number) %>%
  # Group by pitcher and game_date for rolling calculations within each pitcher-game combination
  group_by(pitcher, game_date) %>%
  mutate(
    # Rolling sum of previous 2 at-bats
    rolling_2_at_bats = lag(at_bat_total_score, 1) + lag(at_bat_total_score, 2),
    # Rolling sum of previous 3 at-bats
    rolling_3_at_bats = lag(at_bat_total_score, 1) + lag(at_bat_total_score, 2) + lag(at_bat_total_score, 3)
  ) %>%
  ungroup() %>%
  select(game_date, pitcher, at_bat_number, at_bat_total_score, rolling_2_at_bats, rolling_3_at_bats) %>%
  # Join back to original data using all three keys
  right_join(statcast_25_sp, by = c("game_date", "pitcher", "at_bat_number"))

## Add pitch count
statcast_25_sp <- statcast_25_sp %>%
  arrange(game_date, game_pk, pitcher, at_bat_number) %>% 
  group_by(pitcher, game_pk) %>%
  mutate(pitch_count = row_number()) %>%
  ungroup()

## Determine the fastest pitch type of a certain pitcher
statcast_25_sp <- statcast_25_sp %>% 
  group_by(pitcher, game_pk) %>% 
  mutate(max_any_pitch_type = pitch_type[which.max(release_speed)])

## Add fastest fastball type pitch of each of the at bats
statcast_25_sp <- statcast_25_sp %>%
  group_by(pitcher, game_pk, at_bat_number) %>%
  mutate(
    # fastest FF in PA
    mean_ff = mean(release_speed[pitch_type == max_any_pitch_type], na.rm = TRUE),
    mean_ff = ifelse(is.infinite(mean_ff), NA, mean_ff),
    mean_ff_spin = mean(release_spin_rate[pitch_type == max_any_pitch_type], na.rm = TRUE),
    mean_ff_spin = ifelse(is.infinite(mean_ff_spin), NA, mean_ff_spin), 
    mean_release_point = mean(release_pos_z[pitch_type == max_any_pitch_type], na.rm = TRUE), 
    mean_release_point = ifelse(is.infinite(mean_release_point), NA, mean_release_point)
  ) %>%
  ungroup()

statcast_25_sp <- statcast_25_sp[order(statcast_25_sp$game_date, statcast_25_sp$pitcher, statcast_25_sp$at_bat_number), ]

# yamamoto_opener <- statcast_25_sp %>% 
#   filter(game_date == "2025-03-18")

model_data <- model_data %>% 
  mutate(scored_binary = ifelse(scored > 0, 1, 0))

model_data <- model_data %>% 
  group_by(game_date, pitcher, inning) %>% 
  mutate(predicted_run_expectancy = cumsum(predicted_run_value)) %>%
  ungroup()

model_data <- model_data %>% 
  group_by(game_date, pitcher) %>% 
  mutate(predicted_total_re = cumsum(predicted_run_value)) %>%
  ungroup()

testing_model_data <- testing_model_data %>% 
  mutate(scored_binary = ifelse(scored > 0, 1, 0))

testing_model_data <- testing_model_data %>% 
  group_by(game_date, pitcher, inning) %>% 
  mutate(predicted_run_expectancy = cumsum(predicted_run_value)) %>%
  ungroup()

glm_fit <- glm(
  scored_binary ~ predicted_run_value,
  data = model_data,
  family = binomial
)

summary(glm_fit)

roc_train <- roc(model_data$scored_binary, model_data$predicted_run_value)

best_cutoff <- coords(roc_train, "best", best.method = "closest.topleft")
best_cutoff

model_data$prob <- predict(glm_fit, type = "response")


# Plot ROC curve
roc_curve <- roc(model_data$scored_binary, model_data$prob)
plot(roc_curve, print.thres = TRUE, main = "ROC Curve for Diabetes Prediction")

predicted_class <- ifelse(model_data$predicted_run_value >= best_by_f1$threshold, 1, 0)

# Evaluate model performance
confusion_matrix <- table(model_data$scored_binary, predicted_class)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", accuracy, "\n")
auc(roc_curve)

ggplot(model_data, aes(predicted_run_value, prob)) +
  geom_point(alpha = 0.1) +
  geom_vline(xintercept = best_by_f1$threshold, color = "red") +
  theme_minimal() +
  labs(title = "Classification Boundary Based on Predicted Run Value")


## Yamamoto Opener(Example)
opener <- model_data %>% 
  filter(game_date == "2025-03-18")

opener <- opener %>%
  mutate(inning_change = inning != lag(inning, default = first(inning)))

inning_rows <- which(opener$inning_change)

annot_df <- opener %>%
  dplyr::slice(inning_rows) %>%
  mutate(
    inning_label = inning - 1,       # same logic as your base R text()
    x = at_bat_order,                # where the vertical line goes
    y = 0                             # label y-position
  )

scored_rows <- which(opener$scored > 0)

ggplot(opener) + 
  geom_point(aes(at_bat_order, predicted_run_value), alpha = 0.1) + 
  geom_line(aes(at_bat_order, predicted_run_value, color = "Predicted")) + 
  geom_hline(yintercept = 0.7, color = "red") +
  geom_hline(yintercept = 0, color = "black", alpha = 0.4) +
  geom_point(aes(at_bat_order, run_value), alpha = 0.1) + 
  geom_line(aes(at_bat_order, run_value, color = "Actual"), alpha = 0.4) + 
  scale_color_manual(values = c("Actual" = "gray50", "Predicted" = "black")) +
  geom_vline(
    data = annot_df,
    aes(xintercept = x),
    linetype = "dashed", 
    alpha = 0.4
  ) +
  geom_text(
    data = annot_df,
    aes(x = x, y = y, label = inning_label),
    vjust = 1.2,      # move downward a bit (pos=2 + offset)
    size = 3
  ) +
  geom_point(
    data = opener %>% filter(scored > 0),
    aes(at_bat_order, predicted_run_value),
    color = "red",
    size = 2,
    alpha = 0.9
  ) +
  geom_point(
    data = opener %>% filter(scored > 0),
    aes(at_bat_order, run_value),
    color = "red",
    size = 2,
    alpha = 0.9
  ) +
  theme_classic()


## Random Pitcher
random_pitcher <- model_data %>% 
  filter(pitcher == "676979" & game_date == "2025-08-05")

random_pitcher <- random_pitcher %>%
  mutate(inning_change = inning != lag(inning, default = first(inning)))

inning_rows <- which(random_pitcher$inning_change)

annot_df <- random_pitcher %>%
  dplyr::slice(inning_rows) %>%
  mutate(
    inning_label = inning - 1,       # same logic as your base R text()
    x = at_bat_order,                # where the vertical line goes
    y = 0                             # label y-position
  )

ggplot(random_pitcher) + 
  geom_point(aes(at_bat_order, predicted_run_value), alpha = 0.1) + 
  geom_line(aes(at_bat_order, predicted_run_value, color = "Predicted")) + 
  geom_hline(yintercept = 0.7, color = "red") + 
  geom_point(aes(at_bat_order, run_value), alpha = 0.1) + 
  geom_hline(yintercept = 0, color = "black", alpha = 0.4) +
  geom_line(aes(at_bat_order, run_value, color = "Actual"), alpha = 0.4) + 
  scale_color_manual(values = c("Actual" = "gray50", "Predicted" = "black")) +
  geom_vline(
    data = annot_df,
    aes(xintercept = x),
    linetype = "dashed", 
    alpha = 0.4
  ) +
  geom_text(
    data = annot_df,
    aes(x = x, y = y, label = inning_label),
    vjust = 1.2,      # move downward a bit (pos=2 + offset)
    size = 3
  ) +
  geom_point(
    data = random_pitcher %>% filter(scored > 0),
    aes(at_bat_order, predicted_run_value),
    color = "red",
    size = 2,
    alpha = 0.9
  ) +
  geom_point(
    data = random_pitcher %>% filter(scored > 0),
    aes(at_bat_order, run_value),
    color = "red",
    size = 2,
    alpha = 0.9
  ) +
  theme_classic()


ggplot(random_pitcher) + 
  geom_point(aes(at_bat_order, predicted_run_value), alpha = 0.1) + 
  geom_line(aes(at_bat_order, predicted_run_value, color = "Predicted")) + 
  geom_hline(yintercept = 0.7, color = "red") + 
  geom_hline(yintercept = 0, color = "black", alpha = 0.4) +
  scale_color_manual(values = c("Predicted" = "black")) +
  geom_vline(
    data = annot_df,
    aes(xintercept = x),
    linetype = "dashed", 
    alpha = 0.4
  ) +
  geom_text(
    data = annot_df,
    aes(x = x, y = y, label = inning_label),
    vjust = 1.2,      # move downward a bit (pos=2 + offset)
    size = 3
  ) +
  geom_point(
    data = random_pitcher %>% filter(scored > 0),
    aes(at_bat_order, predicted_run_value),
    color = "red",
    size = 2,
    alpha = 0.9
  ) +
  theme_classic()



# Add some derived metrics
comparison_final <- comparison_proper %>%
  mutate(
    False_Alarm_Rate = round(False_Alarms / N_Triggers * 100, 1),
    Efficiency_Score = Accuracy_Pct / (N_Triggers / 100)  # Accuracy per 100 triggers
  )
comparison_final <- comparison_final[comparison_final$Strategy != "Bases Loaded", ]

# Main visualization: Accuracy vs Trigger Frequency
ggplot(comparison_final, aes(x = N_Triggers, y = Accuracy_Pct)) +
  geom_point(aes(color = Strategy, size = Avg_Runs_After), alpha = 0.7) +
  geom_text(aes(label = Strategy), hjust = -0.1, vjust = 0, size = 3.5) +
  scale_color_manual(values = c("Model-Based" = "#E74C3C", 
                                "Runner 2nd/3rd" = "#3498DB",
                                "Scoring Pos + <2 Outs" = "#2ECC71",
                                "Bases Loaded" = "#F39C12")) +
  scale_size_continuous(range = c(5, 12), name = "Avg Runs\nAfter Trigger") +
  labs(
    title = "Pitching Change Strategy Performance",
    subtitle = "Model triggers less often but with lower accuracy than traditional methods",
    x = "Number of Innings Triggered",
    y = "Accuracy (%)",
    caption = "Ideal position: Top-right (high accuracy, sufficient frequency)\nPoint size represents average runs after trigger"
  ) +
  theme_minimal() +
  theme(legend.position = "right")


analysis_data %>% 
  filter(trad_scenario_1 == "Change Pitcher")

# Prepare data for grouped bar chart
metrics_comparison <- comparison_final %>%
  select(Strategy, Accuracy_Pct, False_Alarm_Rate) %>%
  pivot_longer(cols = -Strategy, names_to = "Metric", values_to = "Value") %>%
  mutate(Metric = recode(Metric, 
                         "Accuracy_Pct" = "Accuracy (%)",
                         "False_Alarm_Rate" = "False Alarm Rate (%)"))

library(knitr)
library(kableExtra)

# Create clean summary
summary_for_slide <- comparison_final %>%
  select(Strategy, N_Triggers, Accuracy_Pct, Avg_Runs_After) %>%
  arrange(desc(Accuracy_Pct)) %>%
  mutate(
    N_Triggers = format(N_Triggers, big.mark = ","),
    Avg_Runs_After = round(Avg_Runs_After, 2)
  ) %>%
  rename(
    "Strategy" = Strategy,
    "Innings Triggered" = N_Triggers,
    "Accuracy %" = Accuracy_Pct,
    "Avg Runs After" = Avg_Runs_After
  )

# Print for viewing
print(summary_for_slide)

# For nice table in RMarkdown/presentation
kable(summary_for_slide, align = 'lrrr') %>%
  kable_styling(bootstrap_options = c("striped", "hover")) %>%
  row_spec(which(summary_for_slide$Strategy == "Model-Based"), 
           bold = TRUE, background = "#FFE5E5")



