# Take prepared data and train model on it
from functions import *
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

reload_config()
from config import *
# Write important variables to config.py:
# How many steps to the future we are going to predict
futureSteps = 10
writevar("futureSteps", futureSteps)
# How many values from the past we are going to use for prediction
pastSteps = futureSteps*5
writevar("pastSteps", pastSteps)
# Time shift to the past; if zero, we are really predicting future data,
# if it equals futureSteps, we are predicting data prior to today.
timeShift = futureSteps+50
writevar("timeShift", timeShift)

# Paths for storing user data
folderPath = os.path.join(os.getenv("APPDATA"), "stopr", ticker)
dataPath = os.path.join(folderPath, "data.csv")
scalerPath = os.path.join(folderPath, "scaler.dat")

# Take "Close" column of data.csv, scale it and save as a new column. Also, save the scaler for future descaling.
df = pd.read_csv(dataPath)
data = df["Close"].to_numpy()

scaler = MinMaxScaler(feature_range=(0.1, 0.9))
scaledData = scaler.fit_transform(data.reshape(-1, 1))
joblib.dump(scaler, scalerPath)

df["Scaled"] = scaledData
# df["Scaled"].round(decimals=2)
df.to_csv(dataPath, index=False)

# Define the training dataset and check if the predictLength is positive
predictLength = int(sum(1 for line in open(dataPath)) - pastSteps)
if predictLength < 0:
    print("Variable *predictLength* isn't positive. Please choose shorter interval.")
trainData = scaledData[:predictLength - pastSteps]
trainData = np.reshape(trainData, (len(trainData), 1))

# In trainX will be *pastSteps* values, and in trainY will be wanted value.
trainX = np.array([])
trainY = np.array([])

for i in range(0, len(trainData) - pastSteps - futureSteps):
    trainX = np.append(trainX, trainData[i:i + pastSteps, 0])
    trainY = np.append(trainY, trainData[i + pastSteps:i + pastSteps + futureSteps, 0])

# Reshape data to 3D, first dimension is count of output data, second defines from how many values it will be
# predicted, and the last dimension is set to 1 (because we have only one type of data)
trainX = np.reshape(trainX, (int(len(trainX) / pastSteps), pastSteps, 1))
trainY = np.reshape(trainY, (int(len(trainY) / futureSteps), futureSteps))

# Build and compile the model, then save it to TICKER dir
model = Sequential()
model.add(LSTM(futureSteps, return_sequences=True, input_shape=(trainX.shape[1], 1)))
model.add(Dropout(0.05))
model.add(LSTM(futureSteps * 2, return_sequences=True))
model.add(Dropout(0.05))
model.add(LSTM(futureSteps * 2))
model.add(Dropout(0.05))
model.add(Dense(futureSteps))
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(trainX, trainY, validation_split=1, epochs=5)
model.save(os.path.join(folderPath, "model.h5"))
