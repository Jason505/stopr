# Take prepared data and train model on it
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Reload config for eventual changes
import importlib
import config
importlib.reload(config)

# Initialization of important global variables:
# How many values from the past we are going to use for prediction:
pastSteps = 120
# How many steps to the future we are going to predict:
futureSteps = 100
# Do we want to predict historic data or *real* future data?
timeShift = futureSteps

# Write variables to config.py
with open("config.py", "a") as f:
    f.write("\npastSteps = " + str(pastSteps))
    f.write("\nfutureSteps = " + str(futureSteps))
    f.write("\ntimeShift = " + str(timeShift))

folderPath = os.path.join(os.getenv("APPDATA"), "Stopr", config.ticker)
dataPath = os.path.join(folderPath, "data.csv")
scalerPath = os.path.join(folderPath, "scaler.dat")

df = pd.read_csv(dataPath)
data = df["Close"]
numpyData = data.to_numpy()

# Scale the dataset and save it to data.csv as a new column, then save the scaler to ticker folder
scaler = MinMaxScaler(feature_range=(0, 1))
scaledData = scaler.fit_transform(numpyData.reshape(-1, 1))
joblib.dump(scaler, scalerPath)

# Save scaled data to data.csv
df["Scaled"] = scaledData
df.to_csv(dataPath, index=False)

# Define the training dataset and check if the predictLength is positive
predictLength = int(sum(1 for line in open(dataPath))-pastSteps)
if predictLength < 0:
    print("Variable *predictLength* isn't positive. Please choose shorter interval.")
trainData = scaledData[:predictLength-pastSteps]
trainData = np.reshape(trainData, (len(trainData), 1))

# In trainX will be *pastSteps* values, and in trainY will be wanted result.
trainX = np.array([])
trainY = np.array([])

for i in range(pastSteps, len(trainData)):
    trainX = np.append(trainX, trainData[i - pastSteps:i, 0])
    trainY = np.append(trainY, trainData[i, 0])

# Convert data sets to more friendly format for LSTM and reshape it to 3D
trainX = np.reshape(trainX, (int(len(trainX)/pastSteps), pastSteps, 1))

# Build and compile the model, then save it to TICKER dir
model = Sequential()
model.add(LSTM(200, return_sequences=True, input_shape=(trainX.shape[1], 1)))
model.add(Dropout(0.1))
model.add(LSTM(100))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(trainX, trainY, validation_split=1, epochs=5, verbose=2)
model.save(os.path.join(folderPath, "model.h5"))
