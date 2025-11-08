import plotly.graph_objs as go
from flask import Flask, render_template
import joblib, pickle, pandas as pd, numpy as np, io, base64, os
import matplotlib
matplotlib.use('Agg')   # non-GUI backend for server
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

# load once at startup
model = joblib.load("best_rf_model.pkl")
scaler = joblib.load("scaler_X.pkl")
with open("feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html", prediction=None, metrics=None, plot_url=None)
@app.route("/predict")
def predict_latest():
    import os, io, base64, math
    import yfinance as yf
    import datetime

    # 1) Load saved metrics (if available)
    metrics = {}
    try:
        metrics_path = os.path.join(os.path.dirname(__file__), "model_metrics.pkl")
        if os.path.exists(metrics_path):
            with open(metrics_path, "rb") as mf:
                metrics = pickle.load(mf) or {}
            # convert to native floats where possible
            for k, v in list(metrics.items()):
                try:
                    metrics[k] = float(v)
                except Exception:
                    pass
        else:
            metrics = {}
    except Exception as e:
        print("Could not load model_metrics.pkl:", e)
        metrics = {}

    # 2) Load latest_features.csv and predict next day
    latest_path = os.path.join(os.path.dirname(__file__), "latest_features.csv")
    if not os.path.exists(latest_path):
        print("latest_features.csv not found at", latest_path)
        return render_template("index.html", prediction=None, metrics=metrics or {}, plot_div=None, plot_url=None, source_info={})

    latest_df = pd.read_csv(latest_path)
    try:
        source_info = latest_df.iloc[0].to_dict()
        # convert numpy types and NaN -> None for template safety
        def clean_record(d):
            out = {}
            for k, v in d.items():
                if pd.isna(v):
                    out[k] = None
                elif isinstance(v, (np.generic,)):
                    out[k] = v.item()
                else:
                    out[k] = v
            return out
        source_info = clean_record(source_info)
    except Exception as e:
        print("Failed to extract source_info:", e)
        source_info = {}

    # Ensure features present and ordered
    X_latest = latest_df.reindex(columns=feature_cols).fillna(0)
    try:
        X_latest_scaled = scaler.transform(X_latest)
        predicted_high = float(model.predict(X_latest_scaled)[0])
    except Exception as e:
        print("Prediction failed on latest_features:", e)
        predicted_high = None

    # 3) Load test_data.csv if available (preferred)
    test_path = os.path.join(os.path.dirname(__file__), "test_data.csv")
    test_df = None
    if os.path.exists(test_path):
        try:
            test_df = pd.read_csv(test_path)
            if "Timestamp" in test_df.columns:
                test_df["Timestamp"] = pd.to_datetime(test_df["Timestamp"])
            print("Loaded test_data.csv:", test_df.shape)
        except Exception as e:
            print("Failed to read test_data.csv:", e)
            test_df = None

    # 4) If no test_data.csv, build from yfinance and safely drop NA only on existing cols
    if test_df is None:
        try:
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=365)
            hist = yf.download("^NSEI", start=start_date, end=end_date, interval="1d")
            hist.reset_index(inplace=True)
            if "Date" in hist.columns and "Timestamp" not in hist.columns:
                hist.rename(columns={"Date": "Timestamp"}, inplace=True)

            # feature engineering (match your training pipeline as closely as possible)
            hist['previousopen'] = hist['Open'].shift(1)
            hist['previousclose'] = hist['Close'].shift(1)
            hist['previoushigh'] = hist['High'].shift(1)
            hist['previousvol'] = hist['Volume'].shift(1)
            hist['dayinweek'] = hist['Timestamp'].dt.dayofweek
            hist['ismonthend'] = hist['Timestamp'].dt.is_month_end
            hist['isquarterend'] = hist['Timestamp'].dt.is_quarter_end
            hist['daysin'] = np.sin(2 * np.pi * hist['dayinweek'] / 7)
            hist['daycos'] = np.cos(2 * np.pi * hist['dayinweek'] / 7)
            hist['dailyreturn'] = hist['Close'].pct_change()
            hist['SMA20'] = hist['Close'].rolling(window=5).mean()
            hist['SMA50'] = hist['Close'].rolling(window=20).mean()

            try:
                import ta
                hist['MACD'] = ta.trend.macd(hist['Close'])
                hist['MACDsignal'] = ta.trend.macd_signal(hist['Close'])
                hist['MACDdiff'] = ta.trend.macd_diff(hist['Close'])
                hist['RSI14'] = ta.momentum.rsi(hist['Close'], window=14)
                hist['CMF20'] = ta.volume.chaikin_money_flow(hist['High'], hist['Low'], hist['Close'], hist['Volume'], window=20)
            except Exception:
                # fallback approximations
                hist['MACD'] = hist['Close'].ewm(span=12, adjust=False).mean() - hist['Close'].ewm(span=26, adjust=False).mean()
                hist['MACDsignal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
                hist['MACDdiff'] = hist['MACD'] - hist['MACDsignal']
                hist['RSI14'] = np.nan
                hist['CMF20'] = 0.0

            hist['EMA7'] = hist['Close'].ewm(span=7, adjust=False).mean()
            hist['EMA14'] = hist['Close'].ewm(span=14, adjust=False).mean()
            hist['targethigh'] = hist['High'].shift(-1)

            # fill na for selected columns (like training)
            for c in ['MACDdiff', 'MACDsignal', 'MACD', 'SMA50', 'CMF20', 'RSI14']:
                if c in hist.columns:
                    hist[c] = hist[c].fillna(hist[c].mean())

            # Option B: only dropna on subset of columns that actually exist in hist
            required = [
                'previousopen','previousclose','previoushigh','previousvol',
                'dayinweek','ismonthend','isquarterend','daysin','daycos',
                'dailyreturn','SMA20','SMA50','MACD','MACDsignal','MACDdiff',
                'EMA7','EMA14','RSI14','CMF20','targethigh'
            ]
            present = [c for c in required if c in hist.columns]
            if present:
                hist = hist.dropna(subset=present)
            else:
                # nothing to drop; proceed (we'll fill missing features later)
                pass

            test_df = hist.copy()
            print("Built test_df from yfinance:", test_df.shape)
        except Exception as e:
            print("Failed to build test_df from yfinance:", e)
            test_df = None

    # 5) If test_df exists, predict across it and build Plotly (interactive) or Matplotlib fallback
    plot_div = None
    plot_url = None
    if test_df is not None and len(test_df) > 0:
        # ensure feature columns exist; fill missing with 0
        X_test = test_df.reindex(columns=feature_cols).fillna(0)
        try:
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
        except Exception as e:
            print("Failed to predict on test set:", e)
            y_pred = None

        if y_pred is not None:
            # actuals
            if 'targethigh' in test_df.columns:
                y_test = test_df['targethigh'].values[:len(y_pred)]
                y_pred = y_pred[:len(y_test)]
            else:
                # fallback: shift High column
                if 'High' in test_df.columns:
                    y_test = test_df['High'].shift(-1).dropna().values
                    minlen = min(len(y_test), len(y_pred))
                    y_test = y_test[:minlen]
                    y_pred = y_pred[:minlen]
                else:
                    y_test = None

            # compute metrics if we have y_test
            if y_test is not None and len(y_test) > 0:
                try:
                    mse = float(mean_squared_error(y_test, y_pred))
                    rmse = float(math.sqrt(mse))
                    r2 = float(r2_score(y_test, y_pred))
                    metrics = {"MSE": mse, "RMSE": rmse, "R2": r2}
                except Exception as e:
                    print("Metrics computation failed:", e)

            # Try Plotly interactive plot first
            try:
                import plotly.graph_objs as go
                if 'Timestamp' in test_df.columns:
                    x = pd.to_datetime(test_df['Timestamp']).iloc[:len(y_pred)]
                    x_label = "Date"
                else:
                    x = np.arange(len(y_pred))
                    x_label = "Samples"

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y_test, mode='lines+markers',
                                         name='Actual', line=dict(color='royalblue', width=2),
                                         marker=dict(size=5)))
                fig.add_trace(go.Scatter(x=x, y=y_pred, mode='lines+markers',
                                         name='Predicted', line=dict(color='orange', width=2, dash='dash'),
                                         marker=dict(symbol='x', size=6)))

                fig.update_layout(title="Actual vs Predicted High Prices",
                                  xaxis_title=x_label, yaxis_title="Price",
                                  legend=dict(x=0.02, y=0.98), template="plotly_white",
                                  height=450, margin=dict(l=40, r=20, t=50, b=40))

                plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn')
            except Exception as e:
                print("Plotly failed, falling back to Matplotlib:", e)
                # Matplotlib fallback
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(12,5), dpi=120)
                    if 'Timestamp' in test_df.columns:
                        x = pd.to_datetime(test_df['Timestamp']).iloc[:len(y_pred)]
                        plt.plot(x, y_test, marker='o', markersize=4, linewidth=1.2, label='Actual', color='tab:blue')
                        plt.plot(x, y_pred, marker='x', markersize=5, linewidth=1.0, linestyle='--', label='Predicted', color='tab:orange')
                        plt.gcf().autofmt_xdate()
                    else:
                        x = np.arange(len(y_test))
                        plt.plot(x, y_test, marker='o', markersize=4, linewidth=1.2, label='Actual', color='tab:blue')
                        plt.plot(x, y_pred, marker='x', markersize=5, linewidth=1.0, linestyle='--', label='Predicted', color='tab:orange')

                    plt.title("Actual vs Predicted High Prices")
                    plt.xlabel("Date" if 'Timestamp' in test_df.columns else "Samples")
                    plt.ylabel("Price")
                    plt.grid(alpha=0.25)
                    plt.legend(loc='upper left', frameon=True)
                    plt.tight_layout()

                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    plt.close()
                    plot_url = f"data:image/png;base64,{plot_b64}"
                except Exception as e:
                    print("Matplotlib fallback also failed:", e)
                    plot_url = None

    # 6) Final safe metrics and render
    safe_metrics = {}
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            try:
                safe_metrics[k] = float(v) if v is not None else None
            except Exception:
                safe_metrics[k] = v

    return render_template("index.html",
                           prediction=(float(predicted_high) if predicted_high is not None else None),
                           metrics=safe_metrics,
                           plot_div=plot_div,
                           plot_url=plot_url,
                           source_info=source_info)


if __name__ == "__main__":
    app.run(debug=True)
