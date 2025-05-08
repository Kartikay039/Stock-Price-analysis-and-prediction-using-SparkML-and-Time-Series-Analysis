from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from sparkxgb import XGBoostRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from sklearn.metrics import median_absolute_error, mean_squared_log_error
import numpy as np
import pandas as pd

# ---------------------- Spark Session ---------------------- #
spark = SparkSession.builder \
    .appName("XGBoost Spark") \
    .config("spark.jars.packages", "ml.dmlc:xgboost4j-spark_2.12:1.6.1") \
    .getOrCreate()

# ---------------------- Load Data ---------------------- #
data_path = r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\AMD_output\part-00000-88d747c0-e1c3-4314-bcd6-a3bf54a570b7-c000.csv"
df = spark.read.option("header", True).option("inferSchema", True).csv(data_path)

# Drop unnecessary columns
columns_to_drop = ["Brand_Name", "Ticker", "Industry_Tag", "Country"]
df = df.drop(*columns_to_drop)

# ---------------------- Feature Engineering ---------------------- #
label_col = 'Close'
feature_cols = [col for col in df.columns if col not in [label_col, "Date"]]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df).select("Date", "features", label_col)

# ---------------------- Train/Test Split ---------------------- #
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# ---------------------- XGBoost Regressor ---------------------- #
xgb = XGBoostRegressor(
    featuresCol="features",
    labelCol=label_col,
    predictionCol="prediction",
    objective="reg:squarederror",
    numRound=100,
    maxDepth=5,
    eta=0.1,
    numWorkers=1
)

# ---------------------- Train Model ---------------------- #
model = xgb.fit(train_data)

# ---------------------- Predict ---------------------- #
predictions = model.transform(test_data)

# ---------------------- Evaluation ---------------------- #
evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction")
rmse = evaluator.setMetricName("rmse").evaluate(predictions)
mse = evaluator.setMetricName("mse").evaluate(predictions)
mae = evaluator.setMetricName("mae").evaluate(predictions)
r2 = evaluator.setMetricName("r2").evaluate(predictions)

# Convert to Pandas for advanced metrics
preds_pd = predictions.select("Date", "prediction", label_col).toPandas()
preds_pd.rename(columns={"prediction": "Forecast", label_col: "Actual"}, inplace=True)
preds_pd = preds_pd.dropna()

# Filter for MLSE (requires positive values)
preds_pd = preds_pd[(preds_pd["Actual"] > 0) & (preds_pd["Forecast"] > 0)]

# Compute additional metrics
mape = np.mean(np.abs((preds_pd["Actual"] - preds_pd["Forecast"]) / preds_pd["Actual"])) * 100
medae = median_absolute_error(preds_pd["Actual"], preds_pd["Forecast"])
try:
    mlse = mean_squared_log_error(preds_pd["Actual"], preds_pd["Forecast"])
except ValueError:
    mlse = np.nan
    print("MLSE could not be computed due to invalid values.")

# ---------------------- Print Metrics ---------------------- #
print(f"[XGBoost] RMSE : {rmse:.4f}")
print(f"          MSE  : {mse:.4f}")
print(f"          MAE  : {mae:.4f}")
print(f"          MAPE : {mape:.2f}%")
print(f"          R²   : {r2:.4f}")
print(f"          MedAE: {medae:.4f}")
print(f"          MLSE : {mlse:.4f}")

# ---------------------- Export Predictions ---------------------- #
output_path = r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\Outputs\xgboost_predictions.csv"
preds_pd.to_csv(output_path, index=False)

# ---------------------- Stop Spark ---------------------- #
spark.stop()


