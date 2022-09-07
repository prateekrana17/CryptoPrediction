import math
import numpy as my_npy
import matplotlib.pyplot as my_plot
import pandas as my_panda
import pandas_datareader as my_reader
import datetime as my_date
import sklearn.metrics

# To scale the data in between 0-1 to make the neural network work better
import sklearn.metrics
from sklearn.preprocessing import MinMaxScaler

# Importing dense,dropout, lstm layers
from keras.layers import Dropout, LSTM, Dense

from keras.models import Sequential

# select cryptocurrency (BTC/ETH/XRP/XMR)
curr_crypto = 'ETH'
# select against currency (USD/GBP/etc.)
curr_against = 'GBP'

# start date
begin_dt = my_date.datetime(2016, 1, 1)
# end date
ending_dt = my_date.datetime(2020, 12, 31)

# get finance data from yahoo api
my_data_set = my_reader.DataReader(f'{curr_crypto}-{curr_against}', 'yahoo', begin_dt, ending_dt)
# print(my_data_set.head())

# Preparing data

# scaling data to values between (0,1)
sclr_minmax = MinMaxScaler(feature_range=(0, 1))
dataset_scld = sclr_minmax.fit_transform(my_data_set['Close'].values.reshape(-1, 1))

# number of days the prediction is based on
prdctn_period = 60

# preparing training data
training_x, training_y = [], []

for t in range(prdctn_period, len(dataset_scld)):
    training_x.append(dataset_scld[t - prdctn_period: t, 0])
    training_y.append(dataset_scld[t, 0])

# converting to a numpy array
training_x, training_y = my_npy.array(training_x), my_npy.array(training_y)
# reshaping
training_x = my_npy.reshape(training_x, (training_x.shape[0], training_x.shape[1], 1))

# Creating the neural network
model = Sequential()

# Adding layers
model.add(LSTM(units=50, return_sequences=True, input_shape=(training_x.shape[1], 1)))
# to prevent over-fitting
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
# to prevent over-fitting
model.add(Dropout(0.2))
model.add(LSTM(units=50))
# to prevent over-fitting
model.add(Dropout(0.2))
# to get a single value which would be the prediction
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')
# Training the model
model.fit(training_x, training_y, epochs=25, batch_size=32)

# Testing the model

# start date for test data
test_begin = my_date.datetime(2021, 1, 1)
# end dae for test data
test_ending = my_date.datetime.now()
# testing data set
testing_data_set = my_reader.DataReader(f'{curr_crypto}-{curr_against}', 'yahoo', test_begin, test_ending)
# getting actual prices
actual_crypto_price = testing_data_set['Close'].values
# obtaining the total dataset from testing and training datasets
total_data_set = my_panda.concat((my_data_set['Close'], testing_data_set['Close']), axis=0)
# model inputs
my_inputs = total_data_set[len(total_data_set) - len(testing_data_set) - prdctn_period:].values
# reshaping the model inputs
my_inputs = my_inputs.reshape(-1, 1)
# scaling down model inputs
my_inputs = sclr_minmax.fit_transform(my_inputs)

testing_x = []

for x in range(prdctn_period, len(my_inputs)):
    testing_x.append(my_inputs[x - prdctn_period:x, 0])

testing_x = my_npy.array(testing_x)

# reshaping to add a 3rd dimension
testing_x = my_npy.reshape(testing_x, (testing_x.shape[0], testing_x.shape[1], 1))

# predicting the price
prdctd_crypto_price = model.predict(testing_x)
# inverse scaling the predicted price to get actual values
prdctd_crypto_price = sclr_minmax.inverse_transform(prdctd_crypto_price)

# mae, mse, rmse calculation

mse = sklearn.metrics.mean_squared_error(actual_crypto_price, prdctd_crypto_price)
print("Mean Square Error: ")
print(mse)
rmse = math.sqrt(mse)
print("\nRoot Mean Square Error: ")
print(rmse)
mae = sklearn.metrics.mean_absolute_error(actual_crypto_price, prdctd_crypto_price)
print("\nMean Absolute Error: ")
print(mae)

# Normalised RMSE
maxP = max(actual_crypto_price)
minP = min(actual_crypto_price)
normalizedRMS = rmse/(maxP-minP)
print("\n Normalized RMSE: ")
print(normalizedRMS)

# plotting the predicted and actual prices
my_plot.plot(actual_crypto_price, color='black', label='Actual Price')
my_plot.plot(prdctd_crypto_price, color='green', label='Predicted Price')
my_plot.title(f'{curr_crypto} Price Prediction')
my_plot.xlabel('Number of Days')
my_plot.ylabel(f'Price in {curr_against}')
my_plot.xlim(0, 100)
my_plot.legend(loc='upper left')
my_plot.savefig("predictionPlot1.png")
my_plot.show()
