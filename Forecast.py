import pandas as pd
import streamlit as st
from pickle import load
import statsmodels.api as sm
import matplotlib.pyplot as plt
import altair as alt

# Load your data_close here
data_close = load(open('data_close.sav', 'rb'))

st.title('Apple Stock Forecasting')

# Input for the number of days
periods = st.number_input('Number of Days', min_value=1)

# Generate date range based on the input
date_range = pd.date_range(start='2020-01-01', periods=periods, freq='B')
date_df = pd.DataFrame(date_range, columns=['Date'])

# Create SARIMA model and forecast
model_sarima_final = sm.tsa.SARIMAX(data_close.Close, order=(2, 1, 0), seasonal_order=(1, 1, 0, 63))
sarima_fit_final = model_sarima_final.fit()
forecast = sarima_fit_final.predict(len(data_close), len(data_close) + periods - 1)
forecast_df = pd.DataFrame(forecast, index=date_df['Date'], columns=['Close'])

# Display the forecasted data
st.write(forecast_df)

# Create and display a line plot using Matplotlib
fig, ax = plt.subplots()
ax.plot(forecast_df.index, forecast_df['Close'])
ax.set_title('Apple Stock Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price')
ax.grid(True)

# Display the Matplotlib figure using st.pyplot
st.pyplot(fig)



