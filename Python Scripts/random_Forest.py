from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from sklearn.metrics import median_absolute_error, mean_squared_log_error
import pandas as pd
import numpy as np

# ---------------------- Start Spark Session ---------------------- #
spark = SparkSession.builder \
    .appName("RandomForestRegressor") \
    .getOrCreate()

# ---------------------- Load Dataset ---------------------- #
data_path = r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\AMD_output\part-00000-88d747c0-e1c3-4314-bcd6-a3bf54a570b7-c000.csv"
df = spark.read.option("header", True).option("inferSchema", True).csv(data_path)

# Drop irrelevant columns
df = df.drop("Brand_Name", "Ticker", "Industry_Tag", "Country")

# ---------------------- Feature Engineering ---------------------- #
label_col = "Close"
feature_cols = [col for col in df.columns if col not in [label_col, "Date"]]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df).select("Date", "features", col(label_col).alias("label"))

# ---------------------- Split Data ---------------------- #
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# ---------------------- Train Model ---------------------- #
rf = RandomForestRegressor(featuresCol="features", labelCol="label", predictionCol="prediction", numTrees=100)
model = rf.fit(train_data)

# ---------------------- Predict ---------------------- #
predictions = model.transform(test_data)

# ---------------------- Evaluation ---------------------- #
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction")
rmse = evaluator.setMetricName("rmse").evaluate(predictions)
mse = evaluator.setMetricName("mse").evaluate(predictions)
mae = evaluator.setMetricName("mae").evaluate(predictions)
r2 = evaluator.setMetricName("r2").evaluate(predictions)

# Convert to Pandas for advanced metrics
preds_pd = predictions.select("Date", "prediction", "label").toPandas()
preds_pd.rename(columns={"label": "Actual", "prediction": "Forecast"}, inplace=True)
preds_pd = preds_pd.dropna()

# Compute MAPE
mape = np.mean(np.abs((preds_pd["Actual"] - preds_pd["Forecast"]) / preds_pd["Actual"])) * 100

# Filter for MSLE (must be positive values)
valid_preds = preds_pd[(preds_pd["Actual"] > 0) & (preds_pd["Forecast"] > 0)]

# Compute advanced metrics
medae = median_absolute_error(preds_pd["Actual"], preds_pd["Forecast"])
try:
    mlse = mean_squared_log_error(valid_preds["Actual"], valid_preds["Forecast"])
except ValueError:
    mlse = np.nan
    print("MSLE could not be computed due to invalid values.")

# ---------------------- Print Metrics ---------------------- #
print(f"[Random Forest] RMSE : {rmse:.4f}")
print(f"                 MSE  : {mse:.4f}")
print(f"                 MAE  : {mae:.4f}")
print(f"                 MAPE : {mape:.2f}%")
print(f"                 R²   : {r2:.4f}")
print(f"                 MedAE: {medae:.4f}")
print(f"                 MSLE : {mlse:.4f}")

# ---------------------- Export to CSV ---------------------- #
output_path = r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\Outputs\random_forest_forecast.csv"
preds_pd.to_csv(output_path, index=False)

# ---------------------- Stop Spark ---------------------- #
spark.stop()


