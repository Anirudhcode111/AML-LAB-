# AML Lab Assignment 3 — Time-Series Regression: Stock Price Prediction

## Student Details

- **Name:** Anirudh Awasthi
- **SAP ID:** 590027460
- **Batch:** Batch-5
- **Course:** Applied Machine Learning (CSAI2017P)

---

## Objective

The objective of this lab is to build supervised features from an ordered stock-price series without leaking future information. The assignment also compares random and chronological data splitting and evaluates whether the model provides a useful directional signal for trading.

---

## Dataset

**AAPL Daily Stock Data**

Daily OHLCV data for Apple Inc. (`AAPL`) was obtained using `yfinance`. Five years of daily historical data were used.

The data was converted into chronological order using the `Date` column before creating time-series features.

---

# A1 — Load and Describe the Series

The AAPL daily stock history was loaded, the date column was parsed, and the observations were sorted in ascending chronological order. The closing price was then plotted against the date.

### Key Code

```python
import yfinance as yf
import pandas as pd

df = yf.download(
    "AAPL",
    period="5y",
    interval="1d",
    auto_adjust=False,
    progress=False
)

df = df.reset_index()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
```
## Closing Price Plot
```python
plt.figure(figsize=(12, 5))

plt.plot(df["Date"], df["Close"])

plt.title("AAPL Closing Price")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.grid(True)

plt.show()
```
## Observation
Unlike LA02, where housing records could be treated independently, stock data follows a strict time order. Shuffling the rows would break this sequence and could introduce future information into the model.

# A2 — Lag Features
Lag-based features were created from the closing price to represent recent price behaviour. The target was defined as the next trading day's closing price.

## Key Code
```python
df_lag = df.copy()

df_lag["Close_lag1"] = df_lag["Close"].shift(1)
df_lag["Close_lag2"] = df_lag["Close"].shift(2)
df_lag["Close_lag3"] = df_lag["Close"].shift(3)

df_lag["Close_rolling5"] = df_lag["Close"].rolling(window=5).mean()

df_lag["Target"] = df_lag["Close"].shift(-1)

df_lag = df_lag.dropna()
```

## Features Used
| Feature          | Description                            |
| ---------------- | -------------------------------------- |
| `Close_lag1`     | Previous trading day's close           |
| `Close_lag2`     | Close from two trading days earlier    |
| `Close_lag3`     | Close from three trading days earlier  |
| `Close_rolling5` | 5-day rolling average of closing price |
| `Target`         | Next trading day's closing price       |

## Observation
The lag features capture recent price behaviour, while the 5-day rolling mean smooths short-term fluctuations. The first few rows are removed because enough previous values are not available to calculate all lag and rolling features.

# B1 — Random Split versus Chronological Split
A `RandomForestRegressor` was trained using two different splitting strategies. The first used a random 80/20 split, while the second kept the final 20% of observations as the test period.

## Key Code
```python
features = [
    "Close_lag1",
    "Close_lag2",
    "Close_lag3",
    "Close_rolling5"
]

X = df_lag[features]
y = df_lag["Target"]
```

## Random Split
```python
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    shuffle=True
)

rf_random = RandomForestRegressor(
    n_estimators=100,
    random_state=0
)

rf_random.fit(X_train_r, y_train_r)

pred_random = rf_random.predict(X_test_r)

mae_random = mean_absolute_error(y_test_r, pred_random)
```

## Chronological Split
```python
split_index = int(len(df_lag) * 0.8)

X_train_c = X.iloc[:split_index]
X_test_c = X.iloc[split_index:]

y_train_c = y.iloc[:split_index]
y_test_c = y.iloc[split_index:]

rf_chrono = RandomForestRegressor(
    n_estimators=100,
    random_state=0
)

rf_chrono.fit(X_train_c, y_train_c)

pred_chrono = rf_chrono.predict(X_test_c)

mae_chrono = mean_absolute_error(y_test_c, pred_chrono)
```

## Results
| Split Method        | Test MAE |
| ------------------- | -------: |
| Random Split        |     4.02 |
| Chronological Split |    26.87 |

## Observation
Random splitting gave a much lower MAE of 4.02 than the chronological split at 26.87.
Random shuffling mixes different time periods into training and testing, causing temporal leakage.
The chronological split uses only past data to predict future prices.
Therefore, its higher MAE gives a more realistic measure of forecasting performance.

# C1 — Would You Trade on It?
The chronological model's predictions were converted into an up/down direction signal and compared with the actual next-day price movement.

## Key Code
```python
actual_direction = np.sign(
    y_test_c.values - df_lag["Close"].iloc[split_index:].values
)

pred_direction = np.sign(
    pred_chrono - df_lag["Close"].iloc[split_index:].values
)

directional_accuracy = np.mean(
    actual_direction == pred_direction
)

print("Directional Accuracy:", directional_accuracy)
print("Directional Accuracy (%):", directional_accuracy * 100)
```
## Result
| Metric               | Value |
| -------------------- | ----: |
| Directional Accuracy |   50% |

## Observation
The model achieved 50% directional accuracy, which is close to random guessing. Hence, the model is not reliable enough to support a trading decision.

## Conclusion
The experiment showed that time order is crucial in stock-price prediction. The chronological split gave a more realistic evaluation, while the 50% directional accuracy showed that the model was not useful as a trading signal.