# streamlit_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

st.set_page_config(layout="wide", page_title="Forecast Dashboard")

page_bg_img = '''
<style>
body {
background: linear-gradient(to right, #a8c0ff, #3f2b96);
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("📈 Forecast Model Evaluation & AMD Time Series EDA")

# Hardcoded file paths
amd_path = r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\Outputs\AMD.csv"
metrics_path = r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\Outputs\Results.xlsx"
forecast_paths = [
    r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\Outputs\ann_tuned.csv",
    r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\Outputs\arima_forecast.csv",
    r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\Outputs\exp_smoothing_forecast.csv",
    r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\Outputs\lstm_forecast.csv",
    r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\Outputs\random_forest_forecast.csv",
    r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\Outputs\var_forecast.csv",
    r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\Outputs\xgboost_predictions.csv"
]

# Load AMD data
amd_df = pd.read_csv(amd_path)
amd_df['Date'] = pd.to_datetime(amd_df['Date'], utc=True).dt.tz_convert(None)
amd_df = amd_df[['Date', 'Close']].dropna()
amd_df.set_index('Date', inplace=True)

st.subheader("📊 AMD Stock Time Series Analysis")
st.line_chart(amd_df['Close'], use_container_width=True)

st.markdown("### 🔍 Summary Statistics")
st.write(amd_df.describe())

st.markdown("### 📈 Histogram & KDE of Close Prices")
fig_hist, ax = plt.subplots()
sns.histplot(amd_df['Close'], kde=True, ax=ax)
st.pyplot(fig_hist)

st.markdown("### 📉 Daily Returns")
amd_df['Log_Returns'] = np.log(amd_df['Close'] / amd_df['Close'].shift(1))
st.line_chart(amd_df['Log_Returns'].dropna())

st.markdown("### 🧮 Rolling Statistics")
roll_mean = amd_df['Close'].rolling(window=30).mean()
roll_std = amd_df['Close'].rolling(window=30).std()
fig_roll, ax = plt.subplots()
ax.plot(amd_df['Close'], label='Close')
ax.plot(roll_mean, label='30D MA')
ax.plot(roll_std, label='30D Std')
ax.legend()
st.pyplot(fig_roll)

st.markdown("### 🔄 Differencing to Remove Trends (First Order)")

# First-order differencing
amd_diff = amd_df['Close'].diff().dropna()

fig_diff, ax = plt.subplots()
ax.plot(amd_df['Close'], label='Original Series', alpha=0.5)
ax.plot(amd_diff, label='1st Order Differenced', color='orange')
ax.set_title("Original vs First-order Differenced Series")
ax.legend()
st.pyplot(fig_diff)

st.markdown("#### Stationarity Tests After Differencing")
result_adf_diff = adfuller(amd_diff)
st.write(f"ADF Statistic (Differenced): {result_adf_diff[0]:.4f}")
st.write(f"ADF p-value (Differenced): {result_adf_diff[1]:.4f}")

result_kpss_diff = kpss(amd_diff, regression='c')
st.write(f"KPSS Statistic (Differenced): {result_kpss_diff[0]:.4f}")
st.write(f"KPSS p-value (Differenced): {result_kpss_diff[1]:.4f}")

if result_adf_diff[1] < 0.05 and result_kpss_diff[1] > 0.05:
    st.success("Differenced series is likely stationary.")
elif result_adf_diff[1] < 0.05:
    st.warning("ADF suggests stationarity after differencing, but KPSS does not fully confirm.")
elif result_kpss_diff[1] > 0.05:
    st.warning("KPSS suggests stationarity after differencing, but ADF does not fully confirm.")
else:
    st.error("Differenced series may still be non-stationary.")


st.markdown("### 🧭 Seasonal Decomposition")
decomp = seasonal_decompose(amd_df['Close'].dropna(), model='additive', period=30)
fig_decomp = decomp.plot()
st.pyplot(fig_decomp)

st.markdown("### 📏 Stationarity Tests")
result_adf = adfuller(amd_df['Close'].dropna())
st.write(f"ADF Statistic: {result_adf[0]:.4f}")
st.write(f"ADF p-value: {result_adf[1]:.4f}")

result_kpss = kpss(amd_df['Close'].dropna(), regression='c')
st.write(f"KPSS Statistic: {result_kpss[0]:.4f}")
st.write(f"KPSS p-value: {result_kpss[1]:.4f}")

if result_adf[1] < 0.05 and result_kpss[1] > 0.05:
    st.success("The time series is likely stationary (confirmed by both tests).")
elif result_adf[1] < 0.05:
    st.warning("ADF suggests stationarity but KPSS indicates non-stationarity.")
elif result_kpss[1] > 0.05:
    st.warning("KPSS suggests stationarity but ADF does not confirm it.")
else:
    st.error("Both tests suggest non-stationarity.")

st.markdown("### 🔁 ACF and PACF Plots")
fig_acf, ax = plt.subplots(1, 2, figsize=(14, 5))
sm.graphics.tsa.plot_acf(amd_df['Close'].dropna(), lags=40, ax=ax[0])
sm.graphics.tsa.plot_pacf(amd_df['Close'].dropna(), lags=40, ax=ax[1])
st.pyplot(fig_acf)

# Load metrics Excel file
metrics_df = pd.read_excel(metrics_path)
st.subheader("📋 Model Evaluation Metrics")
st.dataframe(metrics_df, use_container_width=True)

st.markdown("### 📌 Interpretation")
st.markdown("- **XGBoost** performed best across all metrics (lowest RMSE, MAE, MAPE, highest R²).")
st.markdown("- **Random Forest** also had high R² but higher error metrics.")
st.markdown("- **LSTM** and **ANN** performed moderately well.")
st.markdown("- **ARIMA**, **Exponential Smoothing**, and **VAR** underperformed with negative or low R².")

metric_to_plot = st.selectbox("Select a metric to compare:", metrics_df.columns[1:])
fig_bar = px.bar(metrics_df, x="Model Name", y=metric_to_plot, color="Model Name", title=f"Model Comparison: {metric_to_plot}")
st.plotly_chart(fig_bar, use_container_width=True)

# Load forecasts
st.subheader("📉 Forecast vs Actual Visualization")
all_forecasts = []
for file_path in forecast_paths:
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df['Model'] = file_path.split("\\")[-1].replace('.csv', '')

    if 'arima' in file_path.lower() or 'exp_smoothing' in file_path.lower():
        df = df.tail(30)  # Only 30-day forecast for statistical models
    all_forecasts.append(df)
forecast_df = pd.concat(all_forecasts)

selected_models = st.multiselect("Select Models to Display:", forecast_df['Model'].unique(),
                                 default=forecast_df['Model'].unique())
filtered_df = forecast_df[forecast_df['Model'].isin(selected_models)]

fig = px.line(filtered_df, x='Date', y='Actual', color='Model', line_group='Model', title="Forecast vs Actual")
for model in selected_models:
    fig.add_scatter(x=filtered_df[filtered_df['Model'] == model]['Date'],
                    y=filtered_df[filtered_df['Model'] == model]['Forecast'],
                    mode='lines', name=f'{model} Forecast')
st.plotly_chart(fig, use_container_width=True)

st.markdown("### 💡 Notes:")

