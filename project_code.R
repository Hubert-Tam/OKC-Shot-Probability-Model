#NAME: Hubert Tamayo | NUMBER: 9393

# SETUP
# ==================
library(tidyverse)
library(xgboost)

# Log loss function used throughout for evaluation
log_loss <- function(actual, predicted, eps = 1e-15) {
  predicted <- pmin(pmax(predicted, eps), 1 - eps)
  -mean(actual * log(predicted) + (1 - actual) * log(1 - predicted))
}

# LOAD TRAINING DATA
# ========================
# needed guess_max here - contester4_id/distcont4 are mostly blank
# in the first 1000 rows so readr was mistyping them as logical
train_df <- read_csv("data/training.csv.gz", guess_max = 500000)

# Confirm clean parse (should return 0 rows)
stopifnot(nrow(problems(train_df)) == 0)

# FEATURE ENGINEERING: parse closestdefapproach
# ============================================================
# closestdefapproach is stored as a string like "{20.05,18.82,16.57,13.86}"
# representing the closest defender's distance at 1s, 0.75s, 0.5s, and
# 0.25s before the shot. Split into 4 numeric columns.
approach_split <- train_df$closestdefapproach %>%
  str_remove_all("[{}]") %>%
  str_split_fixed(",", 4) %>%
  apply(2, as.numeric)

colnames(approach_split) <- c("approach_100", "approach_075", "approach_050", "approach_025")
train_df <- cbind(train_df, approach_split)

# TRAIN / VALIDATION SPLIT (by season, not random)
# ============================================================
# splitting by season instead of randomly - wanted to test on a
# season the model hasn't seen at all, closer to what testing.csv.gz will be
model_train <- train_df %>% filter(season_id == "fe055")
validation  <- train_df %>% filter(season_id == "2676a")

# NAIVE BASELINE
# ============================================================
overall_rate <- mean(model_train$outcome)
baseline_predictions <- rep(overall_rate, nrow(validation))
baseline_logloss <- log_loss(validation$outcome, baseline_predictions)

# LOGISTIC REGRESSION MODELS (exploratory / process)
# ============================================================
# These were used to test whether engineered features carried
# linear signal before moving to a tree-based model. Kept here
# for process transparency, not used for the final submission.

logit_basic <- glm(outcome ~ shottype + distance + locationx + locationy +
                     three + contested + num_contesters + closestdefdist +
                     dribblesbefore + shotclock + shooterspeed + gamestate,
                   data = model_train, family = binomial(link = "logit"))
logit_basic_predictions <- predict(logit_basic, newdata = validation, type = "response")
logit_basic_logloss <- log_loss(validation$outcome, logit_basic_predictions)

logit_approach <- glm(outcome ~ shottype + distance + locationx + locationy +
                        three + contested + num_contesters + closestdefdist +
                        dribblesbefore + shotclock + shooterspeed + gamestate +
                        approach_100 + approach_075 + approach_050 + approach_025,
                      data = model_train, family = binomial(link = "logit"))
logit_approach_predictions <- predict(logit_approach, newdata = validation, type = "response")
logit_approach_logloss <- log_loss(validation$outcome, logit_approach_predictions)

model_train <- model_train %>% mutate(defender_distance_change_1s = approach_100 - closestdefdist)
validation  <- validation  %>% mutate(defender_distance_change_1s = approach_100 - closestdefdist)

logit_defender_change <- glm(outcome ~ shottype + distance + locationx + locationy +
                               three + contested + num_contesters + closestdefdist +
                               dribblesbefore + shotclock + shooterspeed + gamestate +
                               approach_100 + approach_075 + approach_050 + approach_025 +
                               defender_distance_change_1s,
                             data = model_train, family = binomial(link = "logit"))
logit_defender_change_predictions <- predict(logit_defender_change, newdata = validation, type = "response")
logit_defender_change_logloss <- log_loss(validation$outcome, logit_defender_change_predictions)

# Leave-one-out shooter shooting ability (shrunk toward overall_rate)
shooter_stats <- model_train %>%
  group_by(shooter_id) %>%
  summarise(shooter_attempts = n(), shooter_makes = sum(outcome), .groups = "drop") %>%
  mutate(shooter_ability = (shooter_makes + 100 * overall_rate) / (shooter_attempts + 100))

model_train <- model_train %>%
  left_join(shooter_stats %>% select(shooter_id, shooter_attempts, shooter_makes), by = "shooter_id") %>%
  mutate(shooter_ability_loo = (shooter_makes - outcome + 100 * overall_rate) /
           (shooter_attempts - 1 + 100),
         shooter_ability = shooter_ability_loo)

validation <- validation %>%
  left_join(shooter_stats %>% select(shooter_id, shooter_ability), by = "shooter_id") %>%
  mutate(shooter_ability = coalesce(shooter_ability, overall_rate))

logit_shooter <- glm(outcome ~ shottype + distance + locationx + locationy +
                       three + contested + num_contesters + closestdefdist +
                       dribblesbefore + shotclock + shooterspeed + gamestate +
                       approach_100 + approach_075 + approach_050 + approach_025 +
                       shooter_ability,
                     data = model_train, family = binomial(link = "logit"))
logit_shooter_predictions <- predict(logit_shooter, newdata = validation, type = "response")
logit_shooter_logloss <- log_loss(validation$outcome, logit_shooter_predictions)

# none of these three features (closing speed, distance change, shooter
# ability) moved the logistic model much -- figured the signal was probably
# nonlinear, which is why I moved to xgboost below. tried adding
# shooter_ability to xgboost too but it overfit hard (early stop went from
# 560 rounds to 103), guessing the loo encoding is too noisy for
# low-attempt shooters. didn't have time to fix the shrinkage properly

# FINAL MODEL: XGBOOST (raw features only, no shooter_ability)
# ============================================================
xgb_formula <- outcome ~ shottype + distance + locationx + locationy +
  three + contested + num_contesters + closestdefdist +
  dribblesbefore + shotclock + shooterspeed + gamestate +
  approach_100 + approach_075 + approach_050 + approach_025

x_train <- model.matrix(xgb_formula, data = model_train)[, -1]
x_valid <- model.matrix(xgb_formula, data = validation)[, -1]

dtrain <- xgb.DMatrix(data = x_train, label = model_train$outcome)
dvalid <- xgb.DMatrix(data = x_valid, label = validation$outcome)

xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.05,
  max_depth = 4,
  min_child_weight = 10,
  subsample = 0.8,
  colsample_bytree = 0.8
)

set.seed(123)
xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 1000,
  evals = list(train = dtrain, validation = dvalid),
  early_stopping_rounds = 30,
  verbose = 1
)

xgb_predictions <- predict(xgb_model, dvalid)
xgb_logloss <- log_loss(validation$outcome, xgb_predictions)

# MODEL SCOREBOARD
# ============================================================
model_results <- tibble(
  model = c(
    "naive baseline",
    "basic logistic regression",
    "logistic + defender approach",
    "logistic + approach + defender change",
    "logistic + shooter_ability",
    "XGBoost (final model)"
  ),
  log_loss = c(
    baseline_logloss, logit_basic_logloss, logit_approach_logloss,
    logit_defender_change_logloss, logit_shooter_logloss, xgb_logloss
  )
)

model_results %>%
  arrange(log_loss) %>%
  mutate(log_loss = sprintf("%.6f", log_loss)) %>%
  print()

# Feature importance for the final model (used in writeup)
xgb_importance <- xgb.importance(model = xgb_model)
print(xgb_importance)
xgb.plot.importance(xgb_importance, top_n = 15, measure = "Gain")
calib_df <- tibble(
  pred = xgb_predictions,
  actual = validation$outcome
) %>%
  mutate(bucket = ntile(pred, 10)) %>%
  group_by(bucket) %>%
  summarise(
    avg_pred = mean(pred),      # what the model claimed, on average, in this bucket
    avg_actual = mean(actual),  # what actually happened, on average, in this bucket
    n = n()
  )

print(calib_df)

ggplot(calib_df, aes(x = avg_pred, y = avg_actual)) +
  geom_point(size = 3, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40") +
  labs(
    title = "Calibration: predicted probability vs. actual make rate",
    x = "Average predicted probability",
    y = "Actual make rate",
    caption = "Points near the dashed diagonal indicate well-calibrated predictions"
  ) +
  xlim(0, 1) + ylim(0, 1) +
  theme_minimal()

distance_check <- validation %>%
  mutate(
    predicted_prob = xgb_predictions,
    distance_bucket = cut(distance, breaks = seq(0, 30, by = 2))
  ) %>%
  group_by(distance_bucket) %>%
  summarise(
    avg_predicted = mean(predicted_prob),
    avg_actual = mean(outcome),
    n = n(),
    .groups = "drop"
  ) %>%
  filter(!is.na(distance_bucket))  # drop any shots beyond 30 ft, rare heaves

print(distance_check)

ggplot(distance_check, aes(x = distance_bucket)) +
  geom_point(aes(y = avg_predicted, color = "Predicted"), size = 3) +
  geom_point(aes(y = avg_actual, color = "Actual"), size = 3) +
  labs(
    title = "Make probability by shot distance: model vs. reality",
    x = "Distance from basket (ft, binned)",
    y = "Make probability",
    color = ""
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# LOAD AND PREPARE TESTING DATA (identical steps to training)
# ============================================================
testing <- read_csv("data/testing.csv.gz", guess_max = 500000)
stopifnot(nrow(problems(testing)) == 0)

test_approach_split <- testing$closestdefapproach %>%
  str_remove_all("[{}]") %>%
  str_split_fixed(",", 4) %>%
  apply(2, as.numeric)

colnames(test_approach_split) <- c("approach_100", "approach_075", "approach_050", "approach_025")
testing <- cbind(testing, test_approach_split)

# ============================================================
# GENERATE PREDICTIONS ON TESTING DATA
# ============================================================
# na.action = na.pass keeps every row intact (does not drop rows with
# NA, e.g. shots with fewer than 4 contesters) -- critical because the
# submission requires predictions for every row in the original order.
# XGBoost natively handles missing values, so this is safe.
xgb_test_formula <- update(xgb_formula, NULL ~ .)  # drop outcome (not present in testing)

x_test <- model.matrix(xgb_test_formula, data = testing, na.action = na.pass)[, -1]

# Sanity check: column names must match training exactly
stopifnot(identical(colnames(x_test), colnames(x_train)))
stopifnot(nrow(x_test) == nrow(testing))

dtest <- xgb.DMatrix(data = x_test)
test_predictions <- predict(xgb_model, dtest)

# BUILD submission.csv
# ============================================================
# building off testing's own row order instead of joining against the
# submission template -- one less place for rows to accidentally shuffle
submission <- tibble(
  shot_id = testing$shot_id,
  make_prob = test_predictions
)

stopifnot(nrow(submission) == nrow(testing))
stopifnot(all(!is.na(submission$make_prob)))

write_csv(submission, "submission.csv")