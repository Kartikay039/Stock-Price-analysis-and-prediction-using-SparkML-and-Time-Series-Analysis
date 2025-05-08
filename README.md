# Stock Price analysis and prediction using SparkML and Time Series Analysis
Introduction 
The stock market has formation, and requires measuring prior price movements, applying statistical and machine learning techniques to create future prices, and then adjusting for price trends predicted. In our world of big data, cutting edge technology such as Apache Spark, shows the computing power to extract and analyze massive amounts of financial data. in this project were applying PySpark to manipulate massive stock data, Spark ML to build prediction models and Power BI/Tableau to create dynamic data visualizations. We ultimately want to build five different time-series models to analyze their prediction accuracy, and deploy the best in a production environment for live predictions.
Objectives 
• Develop different time series, Machine Learning and Deep Learning Models that can be scaled to real time forecasting and prediction.
• Compare model performance based on key metrics such as RMSE, MSE ,MAE, MAPE and R².
• Utilize big data tools for scalable and efficient processing of stock market data.
• Visualise Data and prediction for easier and more effective result analysis using visualisation tools like Powe BI, Tableau or python libraries.

Implementation Methodology
 ![image](https://github.com/user-attachments/assets/921b638b-3b48-4b20-9ad0-7f77791c9439)

Step 1: Data Collection and Preprocessing
•	The dataset used contains data of opening, closing, highs and lows of a number of companies with the data being as latest as up to March 30th.
•	The dataset can be accessed from https://www.kaggle.com/datasets/nelgiriyewithana/world-stock-prices-daily-updating.
•	The relevant data of a company of choice needs to be separated from the rest of the dataset and then checked for missing values and handled accordingly.
Step 2: Exploratory Data Analysis (EDA)
•	Analyze historical stock trends using visualization.
•	Identify seasonality, trends, and anomalies using various tests like ADF Test, KPSS Test, ACF and PACF validation, etc.
Step 3: Model Development
•	Implement different Statistical model based on our inference through testing like ARIMA, SARIMA, VAR, etc.
•	Also work on different machine learning and deep learning solutions like ANN, RNN, Boosting, ALS, etc
Step 4: Model Evaluation and Selection
•	Use metrics like RMSE, MAE, and R² to compare model accuracy.
•	Select the best-performing model for deployment.
Step 5: Deployment & Visualization
•	Use Power BI/Tableau for interactive dashboards showing predictions.
•	Explore Scalable Deployment of the model to enable real-time updates to Stock forecasting.


Our extensive time series forecasting study reveals a distinct performance difference between statistical classical models and machine learning/deep learning approaches for stock price prediction.  The first tests for stationarity i.e. ADF, KPSS, and Ljung-Box tests clearly indicated that the raw sub-daily stock price time series data were non-stationary and relieved significantly autocorrelated. The differencing of our time series data made our points of time series stationary, which presented characteristics of white noise to establish that the classical models applied, such as ARIMA and VAR were appropriate.
![image](https://github.com/user-attachments/assets/84c9931d-0592-412d-862c-6845b2501175)
![image](https://github.com/user-attachments/assets/dfeb019d-fd80-45b5-aaf4-58b942755038)
![image](https://github.com/user-attachments/assets/dba34e02-ea1d-4b24-8cf3-8abdfd4b8cee)


All classical models ARIMA, VAR, and Exponential smoothing, did not produce favourable values for our evaluation metrics despite the theory behind their introductions.  For example ARIMA model returned RMSE of 6.40 and R² of - 1.13 for the stock price data which were poor predictors, respectively, and poor model fit statistics. The same conclusion applies to the evaluation metrics for the Exponential smoothing and VAR model as they returned similarly high RMSE and MAE evaluation metrics and negative R² values and independent noise as they could not account for the complexities of nonlinear behavior of data characteristic of real financial time series.
On the other hand, machine learning and deep learning models exhibited stronger performance with XGBoost clearly leading all models, with an RMSE of 1.27, MAE of 0.45, and an extraordinary R² of 0.9991 indicating near perfection in predictive power and generalization. The LSTM model also was successful (RMSE: 2.96, MAE: 2.25, R²: 0.6031) and demonstrated its ability to learn temporal dependencies in sequential data rather well. The ANN model also performed adequately with a RMSE of 3.20 and R² of 0.5371. This level of predictive strength came after extensive preprocessing.
Random Forest with a relatively strong R² of 0.988, did face a high MAPE of 27.02%, indicating a decreased reliability of its percentage-dependent error measurement, possibly due to overfitting or failure to generalize with fluctuating price patterns.
The visualizations derived from Power BI, provided further evidence of the model's accuracy by allowing interactive analysis for a directed visual inspection for patterns, and confirmed the empirical results. In summary, this comparative analysis demonstrates that
While standard statistical models have limitations especially in terms of noisy and non-linear financial data, modern ML/DL techniques including XGBoost, are clearly more effective. XGBoost's success stands out as a candidate for a real-time, scalable tool for deployment in stock price forecasting applications.
Comparative Analysis of PYSPARK ML Models

![image](https://github.com/user-attachments/assets/943e52cb-02ca-4075-a6b1-3b03724c70a9)

Table 2- PYSPARK Model Error Metrics Parameter Evaluation

XGBOOST is the clear winner: it outperforms all other models in every single comparison
for 7 performance metrics I can compare these forecasting models against and so
considering both their level of accuracy as well as reliability, it is also really should be
among top pick by far from all others in an XBoost vs LSTM battle. LSTM then gives way
to cutting edge models like ARIMA, Exponential Smoothing and VAR in 100% of the
cases whilst outperforming traditional models at 71% & full scale: 57.1% for Random
Forest and 39.3% of ANN. Random Forest gives an R score almost 1 (optimal) which
severely penalizes it with its enormous MAPE in comparison and while beating the
traditional models, XGBOOST and LSTM seem to be still the winner. Then comes ANN
which outperforms all traditional statistical models except for few categories where XGBOOST or LSTM are better. Compared against the traditional models, a true underperformer amongst most measures are ARIMA Exponential and VAR. Conclusion: This comparison affirms that the complex models like XGBOOST and LSTM are much more accurate and stable in making stock forecasting predictions.

![image](https://github.com/user-attachments/assets/43c4e3f7-0027-49b8-82c0-235d37037405)

Conclusion 
The results of the stationarity tests (ADF, KPSS, Ljung-Box), showed that the sub-daily time series at the outset was non-stationary and strongly autocorrelated. After differencing the series, it was found to be stationary before exhibiting white noise features, and could investigate classical time series models like ARIMA and VAR. Classical models, however, could not be performant on advanced noisy financial data because they have built-in assumptions and do not take into account those kind of datasets. In comparison, machine learning and deep learning models proved effective at modelling non-linear trends in the stock price time series, particularly with XGBoost, LSTM, and ANN models. XGBoost was the most successful with the best RMSE (1.26) and R² (0.999), making it capable for real-time and scalable stock price forecasting. The LSTM and ANN were also developed with much success, especially after we incorporated best practice data preprocessing steps before modeling. Importantly, the validation of our comparison study through the use of Power BI dashboards to achieve dynamic visual analytics played a role with our outcome and findings. Our comparable study with multiple models highlights the importance of machine learning methods using statistical tests for time series forecasting.
References 
https://scikitlearn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
https://www.kaggle.com/discussions/general/328156
https://www.datacamp.com/tutorial/arima
https://www.simplilearn.com/exponential-smoothing-for-time-series-forecasting-in-python-article
https://www.sciencedirect.com/topics/engineering/artificial-neural-network
https://mlpills.dev/time-series/stationarity-in-time-series-and-how-to-check-it/
https://www.aptech.com/blog/introduction-to-the-fundamentals-of-vector-autoregressive-models/
https://www.microsoft.com/en-us/power-platform/products/power-bi
https://spark.apache.org/
https://developer.nvidia.com/discover/lstm#:~:text=A%20Long%20short%2Dterm%20memory,cycles%20through%20the%20feedback%20loops/

