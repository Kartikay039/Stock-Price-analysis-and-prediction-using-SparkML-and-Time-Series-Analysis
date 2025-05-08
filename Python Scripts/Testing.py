import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, yule_walker
from statsmodels.stats.diagnostic import acorr_ljungbox
import plotly.graph_objects as go
import plotly.subplots as sp

# Initialize Spark
spark = SparkSession.builder.appName("TimeSeriesCleaner").getOrCreate()

# Load and clean data
df = spark.read.option("header", True).option("inferSchema", True)\
    .csv(r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\AMD_output\part-00000-88d747c0-e1c3-4314-bcd6-a3bf54a570b7-c000.csv")\
    .withColumn("date", to_date(col("date")))\
    .dropna(subset=["date", "ticker", "Close"])\
    .orderBy("date")

# Get unique tickers
tickers = [row['ticker'] for row in df.select("ticker").distinct().collect()]

# Loop through tickers
for ticker in tickers:
    print(f"\n📊 Analyzing Ticker: {ticker}")
    ticker_df = df.filter(col("ticker") == ticker).toPandas()
    ticker_df = ticker_df.sort_values("date").set_index("date")
    ts = ticker_df["Close"].dropna()

    if len(ts) < 50:
        print("⛔ Skipping: Not enough data points.")
        continue

    # Stationarity & randomness checks
    adf_stat, adf_p, *_ = adfuller(ts)
    print(f"ADF Test → Statistic: {adf_stat:.4f}, p-value: {adf_p:.4f}")

    try:
        lb_result = acorr_ljungbox(ts, lags=[10], return_df=True)
        lb_p = lb_result.iloc[0]["lb_pvalue"]
    except:
        lb_p = 1.0

    std_dev = np.std(ts)
    if adf_p > 0.05 and lb_p > 0.05 and std_dev < 1.0:
        print("🧹 Removed: Likely Random/White Noise Series")
        continue  # Skip random series

    print("✅ Signal Detected: Proceeding with Analysis")

    # Plot Time Series
    fig = go.Figure([go.Scatter(x=ts.index, y=ts, name=f'{ticker} Close')])
    fig.update_layout(title=f"{ticker} - Time Series", xaxis_title="Date", yaxis_title="Close Price")
    fig.show()

    # Seasonal Decomposition
    try:
        decomposition = seasonal_decompose(ts, model='multiplicative', period=30)
        fig_decomp = sp.make_subplots(rows=4, cols=1, shared_xaxes=True,
                                      subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
        fig_decomp.add_trace(go.Scatter(x=ts.index, y=decomposition.observed), row=1, col=1)
        fig_decomp.add_trace(go.Scatter(x=ts.index, y=decomposition.trend), row=2, col=1)
        fig_decomp.add_trace(go.Scatter(x=ts.index, y=decomposition.seasonal), row=3, col=1)
        fig_decomp.add_trace(go.Scatter(x=ts.index, y=decomposition.resid), row=4, col=1)
        fig_decomp.update_layout(height=900, title_text=f"{ticker} - Seasonal Decomposition")
        fig_decomp.show()
    except Exception as e:
        print(f"⚠️ Decomposition failed: {e}")

    # KPSS Test
    try:
        kpss_stat, kpss_p, *_ = kpss(ts, regression='c')
        print(f"KPSS Test → Statistic: {kpss_stat:.4f}, p-value: {kpss_p:.4f}")
    except Exception as e:
        print(f"⚠️ KPSS Test failed: {e}")

    # ACF & PACF
    lags = 30
    acf_vals = acf(ts, nlags=lags)
    pacf_vals = pacf(ts, nlags=lags)

    fig_corr = sp.make_subplots(rows=2, cols=1, subplot_titles=("ACF", "PACF"))
    fig_corr.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals), row=1, col=1)
    fig_corr.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals), row=2, col=1)
    fig_corr.update_layout(height=600, title_text=f"{ticker} - ACF & PACF")
    fig_corr.show()

    # Yule-Walker
    try:
        rho, sigma = yule_walker(ts, order=4)
        print(f"Yule-Walker AR(4) → Coeffs: {np.round(rho, 4)}, σ²: {sigma:.4f}")
    except Exception as e:
        print(f"⚠️ Yule-Walker failed: {e}")

    # Ljung-Box already done above
    print(f"Ljung-Box Q(10) → p-value: {lb_p:.4f}")
