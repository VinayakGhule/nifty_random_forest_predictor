# import os
import joblib
import pickle
import pandas as pd
import numpy as np
from openchart import NSEData
import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import ta
import math

end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=730)
nse = NSEData()
nse.download()
df = nse.historical(symbol="Nifty 50", exchange="NSE", start=start_date, end=end_date, interval="1d")
df = df.reset_index()

##Feature engineering

df['previousopen'] = df['Open'].shift(1)
df['previousclose'] = df['Close'].shift(1)
df['previoushigh'] = df['High'].shift(1)  # NEW: for high baseline
df['previousvol'] = df['Volume'].shift(1)
df['dayinweek'] = df['Timestamp'].dt.dayofweek
df['ismonthend'] = df['Timestamp'].dt.is_month_end
df['isquarterend'] = df['Timestamp'].dt.is_quarter_end
df['daysin'] = np.sin(2 * np.pi * df['dayinweek'] / 7)
df['daycos'] = np.cos(2 * np.pi * df['dayinweek'] / 7)
df['dailyreturn'] = df['Close'].pct_change()
df['SMA20'] = df['Close'].rolling(window=5).mean()
df['SMA50'] = df['Close'].rolling(window=20).mean()
df['MACD'] = ta.trend.macd(df['Close'])
df['MACDsignal'] = ta.trend.macd_signal(df['Close'])
df['MACDdiff'] = ta.trend.macd_diff(df['Close'])
df['EMA7'] = df['Close'].ewm(span=7, adjust=False).mean()
df['EMA14'] = df['Close'].ewm(span=14, adjust=False).mean()
df['RSI14'] = ta.momentum.rsi(df['Close'], window=14)
df['CMF20'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=20)

df['targethigh'] = df['High'].shift(-1)

df['MACDdiff'] = df['MACDdiff'].fillna(df['MACDdiff'].mean())
df['MACDsignal'] = df['MACDsignal'].fillna(df['MACDsignal'].mean())
df['MACD'] = df['MACD'].fillna(df['MACD'].mean())
df['SMA50'] = df['SMA50'].fillna(df['SMA50'].mean())
df['CMF20'] = df['CMF20'].fillna(df['CMF20'].mean())
df['RSI14'] = df['RSI14'].fillna(df['RSI14'].mean())

df = df.dropna(subset=[
    'previousopen', 'previousclose', 'previoushigh', 'previousvol', 'dayinweek', 'ismonthend', 'isquarterend',
    'daysin', 'daycos', 'dailyreturn', 'SMA20', 'SMA50', 'MACD', 'MACDsignal', 'MACDdiff',
    'EMA7', 'EMA14', 'RSI14', 'CMF20', 'targethigh'
])

feature_cols = [col for col in df.columns if col not in
                ['Timestamp', 'Open', 'Close', 'targetclose', 'targetopen', 'targethigh', 'targetlow']]
X = df[feature_cols]
y = df['targethigh']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3, random_state=42)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [4, 8, 16, None],
    'min_samples_split': [2, 8, 16],
}
rf_grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)
print("Best RF parameters:", rf_grid.best_params_)
best_rf = rf_grid.best_estimator_

y_test = y_test.fillna(method="ffill")

y_pred = best_rf.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RandomForest MSE: {mse}")
print(f"RandomForest RMSE: {rmse}")
print(f"RandomForest R²: {r2}")

metrics = {"MSE": mse, "RMSE": rmse, "R2": r2}
with open("model_metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

importances = pd.Series(best_rf.feature_importances_, index=X.columns)
print("Top 10 Important Features:")
print(importances.sort_values(ascending=False).head(10))

new_df = df.tail(1)   # latest row
feature_cols = [col for col in new_df.columns if col not in
                ['Timestamp', 'Open', 'Close', 'targetclose', 'targetopen', 'targethigh', 'targetlow']]
new_df_test = new_df[feature_cols]
new_df_test.to_csv("latest_features.csv", index=False)

new_df_scaled = scaler_X.transform(new_df_test)
predicted_high = best_rf.predict(new_df_scaled)
print("Predicted next high:", predicted_high[0])

joblib.dump(best_rf, "best_rf_model.pkl")
joblib.dump(scaler_X, "scaler_X.pkl")
with open("feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

test_df = X_test.copy()
test_df['targethigh'] = y_test.values
# If you have timestamps in original df, keep them (example: test_df['Timestamp'] = df.loc[X_test.index, 'Timestamp'])
test_df.to_csv("test_data.csv", index=False)
print("Saved test_data.csv with shape", test_df.shape)

print("Model, scaler, metrics, and features saved successfully.")