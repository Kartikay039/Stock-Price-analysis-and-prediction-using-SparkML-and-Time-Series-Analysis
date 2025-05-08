import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from sklearn.metrics import median_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Activation
from tensorflow.keras.layers import add, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------- Spark Session ---------------------- #
spark = SparkSession.builder \
    .appName("TCN for Time Series") \
    .getOrCreate()

# ---------------------- Load Data ---------------------- #
df = pd.read_csv(r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\AMD_output\part-00000-88d747c0-e1c3-4314-bcd6-a3bf54a570b7-c000.csv", parse_dates=['Date'])
df = df.sort_values("Date")[['Date', 'Close']].dropna()

# ---------------------- Normalize ---------------------- #
scaler = MinMaxScaler()
df['Close'] = scaler.fit_transform(df[['Close']])

# Feature engineering (optional)
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
df['RSI'] = df['Close'].diff().apply(lambda x: max(x, 0)).rolling(14).mean() / df['Close'].diff().abs().rolling(14).mean()

# Fill any NaN values created by rolling windows
df = df.fillna(method='bfill')

# Select features
features = ['Close', 'SMA_5', 'SMA_20', 'EMA_10', 'RSI']

# ---------------------- Sequence Creation ---------------------- #
def create_multivariate_sequences(data, features, lookback=20):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[features].iloc[i:i + lookback].values)
        y.append(data['Close'].iloc[i + lookback])
    return np.array(X), np.array(y)

lookback = 20
X, y = create_multivariate_sequences(df, features, lookback)

# ---------------------- Train/Test Split ---------------------- #
train_size = len(X) - 25
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# ---------------------- Temporal Convolutional Network (TCN) ---------------------- #
def residual_block(x, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0.2):
    # First Conv layer in residual block
    r = Conv1D(filters=nb_filters, kernel_size=kernel_size,
               dilation_rate=dilation_rate, padding=padding)(x)
    r = BatchNormalization()(r)
    r = Activation('relu')(r)
    r = Dropout(dropout_rate)(r)
    
    # Second Conv layer in residual block
    r = Conv1D(filters=nb_filters, kernel_size=kernel_size,
               dilation_rate=dilation_rate, padding=padding)(r)
    r = BatchNormalization()(r)
    r = Activation('relu')(r)
    r = Dropout(dropout_rate)(r)
    
    # 1x1 conv to match input dimensions if needed
    if x.shape[-1] != nb_filters:
        x = Conv1D(filters=nb_filters, kernel_size=1, padding='same')(x)
    
    # Add the residual connection
    x = add([x, r])
    return x

def build_tcn_model(input_shape, nb_filters=64, kernel_size=3, 
                   nb_stacks=1, dilations=[1, 2, 4, 8, 16], dropout_rate=0.2):
    """Build a Temporal Convolutional Network model."""
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Initial conv layer
    x = Conv1D(filters=nb_filters, kernel_size=1, padding='causal')(x)
    
    # TCN Stack with dilated convolutions
    for stack_i in range(nb_stacks):
        for dilation_rate in dilations:
            x = residual_block(x, dilation_rate, nb_filters, kernel_size, 
                              padding='causal', dropout_rate=dropout_rate)
    
    # Apply global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    
    # Output layer
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# ---------------------- Build and Compile the Model ---------------------- #
model = build_tcn_model(
    input_shape=(X_train.shape[1], X_train.shape[2]), 
    nb_filters=64,
    kernel_size=3,
    dilations=[1, 2, 4, 8, 16, 32],
    dropout_rate=0.2
)

model.compile(
    loss="mse",
    optimizer=Adam(learning_rate=1e-4),
    metrics=["mae"]
)

model.summary()

# ---------------------- Train the Model ---------------------- #
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ---------------------- Plot Training History ---------------------- #
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.title("Mean Absolute Error")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()
plt.tight_layout()
plt.savefig("tcn_training_history.png")
plt.close()

# ---------------------- Predictions ---------------------- #
predictions = model.predict(X_test)

# ---------------------- Evaluation ---------------------- #
# Since we only scaled the 'Close' column
original_close_values = df['Close'].values[-len(y_test):]

# Create a dummy array for inverse transform
y_test_dummy = np.zeros((len(y_test), 1))
y_test_dummy[:, 0] = y_test
y_test_actual = scaler.inverse_transform(y_test_dummy)[:, 0]

pred_dummy = np.zeros((len(predictions), 1))
pred_dummy[:, 0] = predictions.flatten()
predicted = scaler.inverse_transform(pred_dummy)[:, 0]

rmse = np.sqrt(mean_squared_error(y_test_actual, predicted))
mse = mean_squared_error(y_test_actual, predicted)
mae = mean_absolute_error(y_test_actual, predicted)
r2 = r2_score(y_test_actual, predicted)
mape = np.mean(np.abs((y_test_actual - predicted) / y_test_actual)) * 100
medae = median_absolute_error(y_test_actual, predicted)

print(f"RMSE: {rmse:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"MedAE: {medae:.4f}")

# ---------------------- Plot Results ---------------------- #
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(df['Date'].iloc[-len(y_test):], y_test_actual, label="Actual")
plt.plot(df['Date'].iloc[-len(y_test):], predicted, label="Predicted")
plt.title("TCN Model Predictions vs Actual")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test_actual, predicted, alpha=0.5)
plt.plot([min(y_test_actual), max(y_test_actual)], [min(y_test_actual), max(y_test_actual)], 'r--')
plt.title("Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.tight_layout()
plt.savefig("tcn_predictions.png")
plt.close()

# ---------------------- Output for Power BI ---------------------- #
result_df = pd.DataFrame({
    'Date': df['Date'].iloc[-len(y_test):].reset_index(drop=True),
    'Actual': y_test_actual,
    'Forecast': predicted,
    'Error': y_test_actual - predicted,
    'AbsError': np.abs(y_test_actual - predicted)
})
result_df.to_csv(r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\Outputs\tcn_forecast.csv", index=False)

# ---------------------- Feature Importance Analysis (Simple Method) ---------------------- #
def feature_importance(model, X, feature_names):
    """Calculate feature importance by perturbing each feature."""
    baseline_pred = model.predict(X)
    baseline_mse = mean_squared_error(y_test, baseline_pred)
    
    importance = {}
    for i, feature in enumerate(feature_names):
        # Create a copy and shuffle a feature
        X_temp = X.copy()
        np.random.shuffle(X_temp[:, :, i])
        
        # Predict with shuffled feature
        pred_temp = model.predict(X_temp)
        mse_temp = mean_squared_error(y_test, pred_temp)
        
        # Calculate importance
        importance[feature] = (mse_temp - baseline_mse) / baseline_mse * 100
    
    return importance

# Calculate feature importance
feature_imp = feature_importance(model, X_test, features)
for feature, imp in sorted(feature_imp.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {imp:.2f}%")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_imp.keys(), feature_imp.values())
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("% Increase in MSE when Feature is Perturbed")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("tcn_feature_importance.png")
plt.close()

# ---------------------- Spark DataFrame for Prediction ---------------------- #
# Convert test set to Spark DataFrame
test_pd = pd.DataFrame()
for i, feature in enumerate(features):
    for j in range(lookback):
        test_pd[f'{feature}_{j}'] = X_test[:, j, i]

test_sdf = spark.createDataFrame(test_pd)

# Create the VectorAssembler
feature_cols = [f'{feature}_{j}' for feature in features for j in range(lookback)]
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features_vec')

# Transform the test DataFrame with the assembler
assembled_test = assembler.transform(test_sdf)

# Add predictions to Spark DataFrame
result_spark_df = spark.createDataFrame(result_df)
result_spark_df.createOrReplaceTempView("tcn_results")

# Show the results
print("\n----- Spark DataFrame Results -----")
spark.sql("SELECT Date, Actual, Forecast, Error FROM tcn_results ORDER BY Date").show(5)

# Save as parquet for efficiency
result_spark_df.write.mode("overwrite").parquet(r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\Outputs\tcn_results.parquet")

# ---------------------- Stop Spark ---------------------- #
spark.stop()

print("TCN Time Series Forecasting complete!")