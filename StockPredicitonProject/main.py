import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM,Dense,Dropout
import streamlit as st
import matplotlib.pyplot as plt
import datetime

start = datetime.datetime(2014, 1, 1)
end = datetime.datetime(2024, 12, 31)

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker','MSFT')

df = yf.download(user_input, start=start, end=end)
st.subheader('Data from 2014-2024')
st.write(df.describe())

st.title('Closing price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.title('Closing price vs Time Chart with 100 Days Moving Average')
MA100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(MA100)
st.pyplot(fig)


st.title('Closing price vs Time Chart with 200 Days Moving Average')
MA100 = df.Close.rolling(100).mean()
MA200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(MA100)
plt.plot(MA200)
st.pyplot(fig)


#Training and Testing of data

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)

#From Sklearn
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


#Splitting the data into training and testing

x_train =[]
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)

#Load our Model
# model = Sequential()
# model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1],1)))
# model.add(Dropout(0.2))
#
# model.add(LSTM(units=60, activation='relu', return_sequences=True))
# model.add(Dropout(0.3))
#
# model.add(LSTM(units=80, activation='relu', return_sequences=True))
# model.add(Dropout(0.4))
#
# model.add(LSTM(units=120, activation='relu'))
# model.add(Dropout(0.5))
#
# model.add(Dense(units=1))
#
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(x_train, y_train, epochs=50)
model = load_model('keras_model.h5')


last_100_days = data_training.tail(100)
final_model = pd.concat([last_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_model)


#Testing
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)

value = scaler.scale_[0]
y_predictions = model.predict(x_test)
scaler_factor = 1/value
y_predictions = y_predictions * scaler_factor
y_test = y_test * scaler_factor


st.subheader('Original vs Predicted Prices')
fig2 =plt.figure(figsize=(10,6))
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_predictions, label='Predicted Prices', color='red')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction')
plt.legend()
st.pyplot(fig2)



