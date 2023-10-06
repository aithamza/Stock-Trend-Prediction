#from flask import Flask, render_template, request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import pickle

import bt as bt


start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Trend Prediction ')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
#df = bt.get(user_input, start = '2010-01-01', end = '2019-12-31')
#df = data.DataReader('AAPL', 'yahoo', start, end)
import yfinance as yahooFinance


# in order to specify start date and
# end date we need datetime package
import datetime

# startDate , as per our convenience we can modify
startDate = datetime.datetime(2015, 1,1)

# endDate , as per our convenience we can modify
endDate = datetime.datetime(2023, 2, 10)
GI = yahooFinance.Ticker(user_input)

df = pd.DataFrame(GI.history(start=startDate, end=endDate))
# df.index = df.index.strftime('%d/%m/%Y')
df
# Describing data
st.subheader('Data from 2015 - 2023')
st.write(df.describe())

df.index = pd.to_datetime(df.index)
#df.index = df.index.strftime('%d/%m/%Y')
#df.index = pd.to_datetime(df.index)

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


# spliting data to training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# Load my model import pickle
# model = load_model('keras_model.h5')
# model = pickle.load(open('model.pkl','rb'))
'''
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
  x_train.append(data_training_array[i-100:i])
  y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences= True, input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu', return_sequences= True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences= True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1))

model.compile(optimizer= 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs= 50)
# Save the model
pickle.dump(model, open('model.pkl', 'wb'))
'''


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

y_predicted = model.predict(x_test)


scaler = scaler.scale_
scale_factor = scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test* scale_factor



# Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize =(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
