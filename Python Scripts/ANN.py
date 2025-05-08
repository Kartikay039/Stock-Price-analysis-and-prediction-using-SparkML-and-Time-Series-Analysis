import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from sklearn.metrics import median_absolute_error, mean_squared_log_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam
import keras_tuner as kt
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col

# ---------------------- Spark Session ---------------------- #
spark = SparkSession.builder \
    .appName("ANN with PySpark and Keras") \
    .getOrCreate()

# ---------------------- Load Data ---------------------- #
df = pd.read_csv(r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\AMD_output\part-00000-88d747c0-e1c3-4314-bcd6-a3bf54a570b7-c000.csv", parse_dates=['Date'])
df = df.sort_values("Date")[['Date', 'Close']].dropna()

# ---------------------- Normalize ---------------------- #
scaler = MinMaxScaler()
df['Close'] = scaler.fit_transform(df[['Close']])

# ---------------------- Sequence Creation ---------------------- #
def create_sequences(data, lookback=20):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

lookback = 20
X, y = create_sequences(df['Close'].values, lookback)

X = X.reshape(X.shape[0], X.shape[1])  # Flatten for ANN input

# ---------------------- Train/Test Split ---------------------- #
train_size = len(X) - 25
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# ---------------------- Keras Hypermodel ---------------------- #
def build_model(hp):
    model = Sequential()
    
    # Tune the number of units in the first Dense layer
    model.add(Dense(units=hp.Int('units_input', min_value=32, max_value=256, step=32), 
                    activation='relu', input_dim=lookback))
    
    # Tune dropout rate
    model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Add additional Dense layers
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=128, step=32), activation='relu'))
    
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(hp.Float('lr', 1e-4, 1e-2, sampling='LOG')), loss='mse')
    return model

# ---------------------- Hyperparameter Tuning ---------------------- #
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='ann_tuning_dir',
    project_name='amd_ann_tuning'
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Search for the best model
tuner.search(X_train, y_train, epochs=50, validation_split=0.1, callbacks=[early_stop], verbose=1)

# Get the best model found by the tuner
best_model = tuner.get_best_models(num_models=1)[0]

# ---------------------- Train the Best Keras Model ---------------------- #
best_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# ---------------------- Predictions ---------------------- #
predictions = best_model.predict(X_test)

# ---------------------- Evaluation ---------------------- #
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
predicted = scaler.inverse_transform(predictions.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test_actual, predicted))
mse = mean_squared_error(y_test_actual, predicted)
mae = mean_absolute_error(y_test_actual, predicted)
r2 = r2_score(y_test_actual, predicted)
mape = np.mean(np.abs((y_test_actual - predicted) / y_test_actual)) * 100
medae = median_absolute_error(y_test_actual, predicted)
mlse = mean_squared_log_error(y_test_actual, predicted)
print(f"RMSE: {rmse:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"MedAE: {medae:.4f}")
print(f"MLSE: {mlse:.4f}")
# ---------------------- Output for Power BI ---------------------- #
result_df = pd.DataFrame({
    'Date': df['Date'].iloc[-len(y_test):].reset_index(drop=True),
    'Actual': y_test_actual.flatten(),
    'Forecast': predicted.flatten()
})
result_df.to_csv(r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\Outputs\ann_tuned.csv", index=False)

# ---------------------- Spark DataFrame for Prediction ---------------------- #
# Convert test set to Spark DataFrame
test_pd = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(lookback)])
test_sdf = spark.createDataFrame(test_pd)

# Create the VectorAssembler to assemble the features
assembler = VectorAssembler(inputCols=[f'feature_{i}' for i in range(lookback)], outputCol='features_vec')

# Transform the test DataFrame with the assembler
assembled_test = assembler.transform(test_sdf)

# Add predictions from Keras model (non-distributed)
test_sdf = test_sdf.withColumn('prediction', Vectors.dense(predictions.flatten()))

# Show the results
test_sdf.show()

# ---------------------- Stop Spark ---------------------- #
spark.stop()
