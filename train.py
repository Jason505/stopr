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
futureSteps = 5
# How many values from the past we are going to use for prediction
pastSteps = futureSteps*5
# Time shift to the past; if zero, we are really predicting future data,
# if it equals futureSteps, we are predicting data prior to today.
timeShift = futureSteps+50

if devMode:
    writevar("futureSteps", futureSteps)
    writevar("pastSteps", pastSteps)
    writevar("timeShift", timeShift)
else:
    writevar("futureSteps")
    writevar("pastSteps")
    writevar("timeShift")

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

print("Model training in progress . . .")
if modelType == 1:
    for i in range(0, len(trainData) - pastSteps - futureSteps):
        trainX = np.append(trainX, trainData[i:i + pastSteps, 0])
        trainY = np.append(trainY, trainData[i + pastSteps:i + pastSteps + futureSteps, 0])
    trainX = np.reshape(trainX, (int(len(trainX) / pastSteps), pastSteps, 1))
    trainY = np.reshape(trainY, (int(len(trainY) / futureSteps), futureSteps, 1))

elif modelType == 2 or modelType == 3:
    for i in range(0, len(trainData) - pastSteps - futureSteps):
        trainX = np.append(trainX, trainData[i:i+pastSteps, 0])
        trainY = np.append(trainY, trainData[i + pastSteps:i + pastSteps + 1, 0])
    trainX = np.reshape(trainX, (int(len(trainX) / pastSteps), pastSteps, 1))
    trainY = np.reshape(trainY, (len(trainY), 1, 1))

if modelType == 4:
    for n in range(0, futureSteps):
        for i in range(0, len(trainData) - pastSteps - futureSteps):
            trainX = np.append(trainX, trainData[i:i + pastSteps, 0])
            trainY = np.append(trainY, trainData[i + pastSteps + n:i + pastSteps + n + 1, 0])
    trainX = np.reshape(trainX, (int(len(trainX) / (pastSteps*futureSteps)), pastSteps, futureSteps))
    trainY = np.reshape(trainY, (int(len(trainY) / futureSteps), 1, futureSteps))

# Build and compile the model, then save it to TICKER dir
model = Sequential()
model.add(LSTM(futureSteps*2, return_sequences=True, input_shape=(trainX.shape[1], 1)))
model.add(Dropout(0.05))
model.add(LSTM(futureSteps * 3, return_sequences=True))
model.add(Dropout(0.05))
model.add(LSTM(futureSteps * 2))
model.add(Dense(trainY.shape[1]))
model.compile(optimizer="adam", loss="mean_squared_error")
for i in range(0, trainX.shape[2]):
    usedX = np.reshape(trainX[:, :, i], (trainX.shape[0], trainX.shape[1], 1))
    usedY = np.reshape(trainY[:, :, i], (trainY.shape[0], trainY.shape[1]))
    if modelType == 4:
        print("\nTraining model for " + str(n+1) + ". step")
    model.fit(usedX, usedY, validation_split=1, epochs=5)
    if modelType != 4:
        model.save(os.path.join(folderPath, "models", "model.h5"))
    else:
        model.save(os.path.join(folderPath, "models", "model_" + str(i) + ".h5"))
