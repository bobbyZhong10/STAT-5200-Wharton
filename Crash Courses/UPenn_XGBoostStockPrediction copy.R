# This project builds a simplified quantitative equity trading system using machine
# learning to forecast relative stock performance and construct a systematic equity
# portfolio. The code collects daily price data for a diversified universe of U.S.
# stocks and a market benchmark (the S&P 500 ETF, SPY). It then engineers predictive
# signals commonly used in quantitative finance, including momentum, volatility,
# short-term reversal, and market-adjusted returns. These features are designed to
# capture systematic patterns in returns rather than relying on individual company
# narratives.
#
# A gradient-boosted decision tree model (XGBoost) is trained using historical data
# and evaluated on a strictly out-of-sample test period to avoid look-ahead bias.
# The model predicts future 20-day returns. During the test period, stocks are
# ranked by predicted performance and the highest-ranked securities are selected
# for a portfolio. Performance is evaluated against a passive S&P 500 benchmark
# using cumulative returns, Sharpe ratio, CAPM beta, and estimated alpha.
#
# The final graph compares the cumulative performance of the machine learning
# portfolio against a passive investment in SPY. This project demonstrates a
# complete quantitative workflow including data acquisition, feature engineering,
# machine learning prediction, portfolio construction, and performance evaluation.
# A real-world implementation would require additional considerations including
# transaction costs, liquidity constraints, rolling model updates, risk controls,
# and robustness testing across different market environments.


# ============================================================
# 1. LIBRARIES
# ============================================================

library(tidyverse)
library(tidyquant)
library(tidymodels)
library(lubridate)
library(slider)
library(xgboost)

set.seed(123)


# ============================================================
# 2. GET REAL STOCK DATA
# ============================================================

# ============================================================
# STOCK UNIVERSE
# ============================================================

# A diversified universe of 100 large U.S. companies
# across technology, finance, healthcare, consumer, industrials,
# energy, and communication sectors.

tickers <- c(
  
  # Technology
  "AAPL","MSFT","NVDA","GOOG","META","AMZN","AVGO","ORCL","ADBE","CRM",
  "CSCO","AMD","INTC","QCOM","TXN","IBM","NOW","INTU","AMAT","MU",
  
  # Financials
  "JPM","BAC","WFC","GS","MS","BLK","C","SCHW","SPGI","AXP",
  "USB","PNC","TFC","BK","COF",
  
  # Healthcare
  "UNH","JNJ","LLY","PFE","MRK","ABBV","TMO","DHR","ABT","BMY",
  "AMGN","GILD","CVS","ISRG","MDT",
  
  # Consumer Discretionary
  "TSLA","HD","MCD","NKE","SBUX","LOW","TJX","BKNG","CMG","ORLY",
  
  # Consumer Staples
  "WMT","COST","PG","KO","PEP","PM","MO","CL","EL",
  
  # Industrials
  "CAT","GE","HON","UPS","BA","RTX","DE","LMT","MMM","ETN",
  
  # Energy
  "XOM","CVX","COP","SLB","EOG",
  
  # Communication / Media
  "NFLX","DIS","CMCSA","TMUS","VZ",
  
  # Real Estate / Utilities / Materials
  "AMT","PLD","NEE","DUK","SO","LIN","APD"
)


prices <- tq_get(
  tickers,
  from = "2022-01-01",
  to = Sys.Date()
)


# ============================================================
# 3. MARKET BENCHMARK
# ============================================================

# SPY is used as a proxy for the S&P 500.

spy <- tq_get(
  "SPY",
  from = "2022-01-01",
  to = Sys.Date()
) %>%
  select(date, spy_price = adjusted) %>%
  mutate(
    spy_ret = spy_price / lag(spy_price) - 1
  )


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

# Create predictive signals commonly used in quantitative finance.

df <- prices %>%
  
  group_by(symbol) %>%
  arrange(date) %>%
  
  mutate(
    # Daily return
    return = adjusted / lag(adjusted) - 1
  ) %>%
  
  ungroup() %>%
  
  left_join(
    spy %>% select(date, spy_ret),
    by = "date"
  ) %>%
  
  group_by(symbol) %>%
  
  mutate(
    
    # Momentum signals
    # Medium-term price trends
    
    mom_1m = lag(adjusted, 21) / lag(adjusted, 1) - 1,
    
    mom_3m = lag(adjusted, 63) / lag(adjusted, 1) - 1,
    
    mom_6m = lag(adjusted, 126) / lag(adjusted, 1) - 1,
    
    
    # Volatility measure
    # Lagged to ensure only past information is used
    
    vol_1m = slide_dbl(
      lag(return),
      sd,
      .before = 21,
      .complete = TRUE
    ),
    
    
    # Short-term reversal
    
    ret_5d = lag(adjusted, 5) / lag(adjusted, 1) - 1,
    
    
    # Market-adjusted return
    
    excess_ret = return - spy_ret
  ) %>%
  
  ungroup() %>%
  
  drop_na()



# ============================================================
# 5. CREATE TARGET VARIABLE
# ============================================================

# Predict the return earned after the portfolio is formed.
# The next-day adjustment avoids using same-day future information.

df <- df %>%
  
  group_by(symbol) %>%
  
  mutate(
    target = lead(adjusted, 20) / lead(adjusted, 1) - 1
  ) %>%
  
  ungroup() %>%
  
  drop_na()



# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

# Use a time-based split to avoid look-ahead bias.

train <- df %>%
  filter(date < "2024-01-01")

test <- df %>%
  filter(date >= "2024-01-01")



# ============================================================
# 7. PREPROCESSING
# ============================================================

# Tree-based models do not require feature normalization.

rec <- recipe(target ~ ., data = train) %>%
  update_role(
    date,
    symbol,
    new_role = "id"
  )



# ============================================================
# 8. MODEL (XGBOOST)
# ============================================================

# Gradient boosting captures nonlinear relationships
# between signals and future returns.

xgb_spec <- boost_tree(
  trees = 300,
  tree_depth = 5,
  learn_rate = 0.05,
  loss_reduction = 0,
  min_n = 10
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")



# ============================================================
# 9. TRAIN FINAL MODEL
# ============================================================

final_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(xgb_spec)


final_fit <- fit(
  final_wf,
  data = train
)



# ============================================================
# 10. GENERATE PREDICTIONS
# ============================================================

preds <- predict(final_fit, test) %>%
  bind_cols(test)

# ============================================================
# 11. CONSTRUCT CURRENT PORTFOLIO
# ============================================================

# Select the most recent prediction date.

latest_date <- max(preds$date)

latest_preds <- preds %>%
  filter(date == latest_date) %>%
  arrange(desc(.pred))


# Select the 10 highest-ranked stocks.

portfolio <- latest_preds %>%
  slice(1:10) %>%
  mutate(
    
    # Remove negative predictions
    pred_positive = pmax(.pred, 0),
    
    
    # Normalize portfolio weights
    weight = ifelse(
      sum(pred_positive) == 0,
      1 / n(),
      pred_positive / sum(pred_positive)
    )
  )


print(portfolio)



# ============================================================
# 12. BACKTEST THE STRATEGY
# ============================================================

# Rebalance monthly because the model predicts 20-day returns.

backtest <- preds %>%
  
  mutate(
    month = floor_date(date, "month")
  ) %>%
  
  group_by(month) %>%
  
  filter(date == min(date)) %>%
  
  arrange(desc(.pred)) %>%
  
  slice(1:10) %>%
  
  mutate(
    
    pred_positive = pmax(.pred, 0),
    
    weight = ifelse(
      sum(pred_positive) == 0,
      1/n(),
      pred_positive / sum(pred_positive)
    )
    
  ) %>%
  
  ungroup()



# ============================================================
# 13. COMPUTE PORTFOLIO RETURNS
# ============================================================

# Calculate hypothetical portfolio performance.
# This evaluates historical model predictions.

returns <- backtest %>%
  
  group_by(month, date) %>%
  
  summarise(
    portfolio_return = sum(weight * target, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  
  mutate(
    cumulative_return = cumprod(1 + portfolio_return)
  )



# ============================================================
# 14. BENCHMARK (SPY)
# ============================================================

spy_returns <- spy %>%
  
  mutate(
    month = floor_date(date, "month")
  ) %>%
  
  group_by(month) %>%
  
  summarise(
    spy_return = prod(1 + (spy_price / lag(spy_price) - 1), na.rm = TRUE) - 1,
    .groups = "drop"
  ) %>%
  
  mutate(
    spy_cumulative = cumprod(1 + spy_return)
  )


returns <- returns %>%
  
  left_join(
    spy_returns,
    by = "month"
  )



# ============================================================
# 15. PERFORMANCE METRICS
# ============================================================


# Annualized Sharpe ratio

sharpe <- mean(
  returns$portfolio_return,
  na.rm = TRUE
) /
  sd(
    returns$portfolio_return,
    na.rm = TRUE
  ) *
  sqrt(252)


cat(
  "Annualized Sharpe Ratio:",
  round(sharpe, 2),
  "\n"
)



# CAPM regression

capm <- lm(
  portfolio_return ~ spy_return,
  data = returns
)


alpha <- coef(capm)[1] * 252

beta <- coef(capm)[2]


cat(
  "Annualized Alpha:",
  round(alpha, 4),
  "\n"
)

cat(
  "Beta:",
  round(beta, 3),
  "\n"
)



# ============================================================
# 16. CUMULATIVE RETURNS GRAPH
# ============================================================

ggplot(
  returns,
  aes(date)
) +
  
  geom_line(
    aes(
      y = cumulative_return,
      color = "Machine Learning Portfolio"
    ),
    linewidth = 1.2
  ) +
  
  geom_line(
    aes(
      y = spy_cumulative,
      color = "S&P 500 (SPY)"
    ),
    linewidth = 1.2
  ) +
  
  labs(
    title = "Machine Learning Portfolio vs. S&P 500",
    subtitle = "Out-of-sample backtest",
    x = "Date",
    y = "Growth of $1",
    color = ""
  ) +
  
  theme_minimal()



# ============================================================
# 17. CURRENT MODEL RANKING
# ============================================================

recommended_portfolio <-
  
  preds %>%
  
  filter(date == max(date)) %>%
  
  arrange(desc(.pred)) %>%
  
  mutate(
    
    Rank = row_number(),
    
    predicted_return =
      round(.pred * 100, 2),
    
    
    pred_positive =
      pmax(.pred, 0),
    
    
    weight =
      ifelse(
        sum(pred_positive) == 0,
        1 / n(),
        pred_positive / sum(pred_positive)
      ),
    
    
    weight =
      round(weight * 100, 2)
    
  ) %>%
  
  select(
    Rank,
    Ticker = symbol,
    `Predicted Return (%)` = predicted_return,
    `Portfolio Weight (%)` = weight
  )


cat("\n")
cat("========================================\n")
cat(" Full Model Stock Ranking\n")
cat("========================================\n")
cat(
  "Generated on:",
  as.character(Sys.Date()),
  "\n"
)
cat("========================================\n\n")


print(recommended_portfolio, n = Inf)


