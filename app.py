import pandas_datareader as data
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st

end_date=datetime.today().date()
start_date= (end_date - pd.DateOffset(years=10)).date()

st.title('Stock Trend Prediction')
user_input=st.text_input('Enter Stock Ticker','AAPL')

df= yf.download(user_input,start_date,end_date)

#describing the data
st.subheader('Data from 2015-2025')
st.write(df.describe())

df= df.reset_index()
df=df.drop(['Date'],axis=1)

#visualize the data
st.subheader("Closing Price vs Time Chart")
fig=plt.figure(figsize=(12,6))
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100MA")
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100MA & 200MA")
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df['Close'])
st.pyplot(fig)

#splitting data into training and testing

data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)

#scale down the data
scaler=MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(data_training)

x_train=[]
y_train=[]


#splitting the training data

for i in range(100,data_training_array.shape[0]):
  x_train.append(data_training_array[i-100:i])
  y_train.append(data_training_array[i,0])

x_train,y_train=np.array(x_train),np.array(y_train)

#model 
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

if os.path.exists('stock_model.keras'):
    model = load_model('stock_model.keras')
else:
   model=Sequential()
   
   model.add(LSTM(units=50,activation='relu',return_sequences=True,input_shape=(x_train.shape[1],1)))
   model.add(Dropout(0.2))

   model.add(LSTM(units=60,activation='relu',return_sequences=True))
   model.add(Dropout(0.3))

   model.add(LSTM(units=80,activation='relu',return_sequences=True))
   model.add(Dropout(0.4))

   model.add(LSTM(units=120,activation='relu'))
   model.add(Dropout(0.5))

   model.add(Dense(units=1))

   model.compile(optimizer='adam',loss='mean_squared_error')
   model.fit(x_train,y_train,epochs=20)
   model.save('stock_model.keras')

past_100_days=data_training.tail(100)
final_df=pd.concat([past_100_days, data_testing], ignore_index=True)

input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

#making a prediction
y_predicted=model.predict(x_test)

#Scale up the data 
scale_factor=1/0.00741717
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

#visualize the prediction
st.subheader("Original Closing Price vs Predicted Price")
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
