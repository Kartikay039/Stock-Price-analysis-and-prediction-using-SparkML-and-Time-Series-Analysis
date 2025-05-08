import os
os.environ['PYSPARK_PYTHON'] = r"C:\Users\melio\AppData\Local\Programs\Python\Python310\python.exe"
os.environ['PYSPARK_DRIVER_PYTHON'] = r"C:\Users\melio\AppData\Local\Programs\Python\Python310\python.exe"

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
from tensorflow.keras.optimizers.legacy import Adam  # Using legacy Adam for Elephas compatibility
import tensorflow as tf

# Enable eager execution for TensorFlow (sometimes helps with Elephas compatibility)
# Using the correct deprecation warning from the log
tf.compat.v1.enable_eager_execution()

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("StockPricePrediction") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.cores", "2") \
    .config("spark.task.cpus", "1") \
    .config("spark.python.worker.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to reduce verbose output
spark.sparkContext.setLogLevel("ERROR")

try:
    # Load data using Pandas
    df = pd.read_csv(r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\AMD_output\part-00000-88d747c0-e1c3-4314-bcd6-a3bf54a570b7-c000.csv", parse_dates=['Date'])
    df = df.sort_values("Date")[['Date', 'Close']].dropna()

    # Normalize the 'Close' prices
    scaler = MinMaxScaler()
    close_values = df[['Close']].values  # Get values as 2D array
    df['Close'] = scaler.fit_transform(close_values)

    # Create sequences
    def create_sequences(data, lookback=20):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i + lookback])
            y.append(data[i + lookback])
        return np.array(X), np.array(y)

    lookback = 20
    X, y = create_sequences(df['Close'].values, lookback)

    # Train/test split
    train_size = int(len(X) * 0.8)  # Use 80% of data for training instead of fixed size
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Define Keras LSTM model
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(lookback, 1)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')  # Using legacy Adam optimizer
    
    # Print model summary
    model.summary()

    # Convert data to RDD - Don't flatten the data as it needs to maintain its shape
    rdd = to_simple_rdd(spark.sparkContext, X_train, y_train)

    # Wrap Keras model with Elephas SparkModel
    spark_model = SparkModel(model, 
                            frequency='epoch', 
                            mode='synchronous',
                            num_workers=1)  # Limit number of workers

    # Train distributed
    spark_model.fit(rdd, epochs=1, batch_size=8, verbose=1, validation_split=0.1)

    # Evaluate the model - using the keras model directly for predictions
    predicted = model.predict(X_test)  # Use the keras model directly
    
    # Inverse transform for actual values
    predicted_reshaped = predicted.reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)
    
    predicted_actual = scaler.inverse_transform(predicted_reshaped)
    y_test_actual = scaler.inverse_transform(y_test_reshaped)

    # Evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test_actual, predicted_actual))
    mse = mean_squared_error(y_test_actual, predicted_actual)
    mae = mean_absolute_error(y_test_actual, predicted_actual)
    r2 = r2_score(y_test_actual, predicted_actual)
    mape = np.mean(np.abs((y_test_actual - predicted_actual) / y_test_actual)) * 100

    print(f"RMSE: {rmse:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # Prepare result DataFrame for Power BI
    result_df = pd.DataFrame({
        'Date': df['Date'].iloc[-len(y_test):].reset_index(drop=True),
        'Actual': y_test_actual.flatten(),
        'Forecast': predicted_actual.flatten()
    })
    
    # Create output directory if it doesn't exist
    output_dir = r"C:\Users\melio\OneDrive\Desktop\labs and lectures\data-science\big data\Spark\Outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    result_path = os.path.join(output_dir, "lstm_distributed.csv")
    model_path = os.path.join(output_dir, "lstm_model")
    
    result_df.to_csv(result_path, index=False)
    
    # Save the Keras model rather than the Spark model
    model.save(model_path)
    
    print(f"Results saved to {result_path}")
    print(f"Model saved to {model_path}")

except Exception as e:
    print(f"Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()

finally:
    # Stop SparkSession
    spark.stop()
