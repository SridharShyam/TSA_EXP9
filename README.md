# EX.NO.09 - A project on Time series analysis on weather forecasting using ARIMA model 
#### Name: SHYAM S
#### Reg.No: 212223240156

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
data=pd.read_csv("/content/index_1.csv")
data['datetime'] = pd.to_datetime(data['datetime'])

data.head()
data.tail()

daily_sales = data.groupby('date')['money'].sum().reset_index()
daily_sales['date'] = pd.to_datetime(daily_sales['date'])
daily_sales.set_index('date', inplace=True)

def arima_model(data, target_variable, order):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]
    
    # Fit ARIMA model
    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()
    
    # Forecast for the test period
    forecast = fitted_model.forecast(steps=len(test_data))
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for Daily ' + target_variable.capitalize())
    plt.legend()
    plt.show()
    
    print("Root Mean Squared Error (RMSE):", rmse)
  arima_model(daily_sales, 'money', order=(5,1,0))
```
### OUTPUT:
<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/7441f876-0894-42ec-8027-d9350ceeee2a" />

<img width="497" height="43" alt="image" src="https://github.com/user-attachments/assets/17ee809c-f18d-4652-b821-23886700cbbf" />

### RESULT:
Thus the program run successfully based on the ARIMA model using python.
