import streamlit as st
import pickle
import pandas as pd
import yfinance as yahooFinance
import numpy as np
import datetime
import matplotlib.pyplot as plt

# startDate , as per our convenience we can modify
startDate = datetime.datetime(2015, 1,1)
endDate = datetime.datetime(2023, 2, 10)
st.title('Stock Trend Prediction ')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
startDate = st.date_input('Enter Start Date', startDate)
endDate = st.date_input('Enter End Date', endDate)
# endDate , as per our convenience we can modify

GI = yahooFinance.Ticker(user_input)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

# spliting data to training and testing
df = pd.DataFrame(GI.history(start=startDate, end=endDate))

# Visualisations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize= (12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize= (12,6))
plt.plot(df.Close)
plt.plot(ma100)
st.pyplot(fig)


data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], axis=0)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Making Predections
model = pickle.load(open("model.pkl", "rb"))
y_predicted = model.predict(x_test)


scaler = scaler.scale_
scale_factor = scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor



# Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize =(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
