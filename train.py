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
pastSteps = 150
# How many steps to the future we are going to predict:
futureSteps = 30
# Do we want to predict historic data or *real* future data?
timeShift = futureSteps+50

# Define function for writing variables
def writeVar(var):
    try:
        config.var # TODO Fix referencing right value, not "var"
    except NameError:
        print("Enter value of \'" + var + "\":")
        userInput = input()
        varWritten = False
        while not varWritten:
            #try:
                if int(varWritten) != 0:
                    print("TADA")


# Write variables to config.py
with open("config.py", "a") as f:
    try:
        config.pastSteps
    except (NameError, AttributeError) as e:
        f.write("\npastSteps = " + str(pastSteps))
    try:
        config.futureSteps
    except (NameError, AttributeError) as e:
        f.write("\nfutureSteps = " + str(futureSteps))
    try:
        config.timeShift
    except (NameError, AttributeError) as e:
        f.write("\ntimeShift = " + str(timeShift))

# Paths for storing user data
folderPath = os.path.join(os.getenv("APPDATA"), "stopr", config.ticker)
dataPath = os.path.join(folderPath, "data.csv")
scalerPath = os.path.join(folderPath, "scaler.dat")

# Take "Close" column of data.csv, scale it and save as a new column. Also, save the scaler for future descaling.
df = pd.read_csv(dataPath)
data = df["Close"].to_numpy()

scaler = MinMaxScaler(feature_range=(0.25, 0.75))
scaledData = scaler.fit_transform(data.reshape(-1, 1))
joblib.dump(scaler, scalerPath)

df["Scaled"] = scaledData
#df["Scaled"].round(decimals=2)
df.to_csv(dataPath, index=False)

# Define the training dataset and check if the predictLength is positive
predictLength = int(sum(1 for line in open(dataPath))-pastSteps)
if predictLength < 0:
    print("Variable *predictLength* isn't positive. Please choose shorter interval.")
trainData = scaledData[:predictLength-pastSteps]
trainData = np.reshape(trainData, (len(trainData), 1))

# In trainX will be *pastSteps* values, and in trainY will be wanted value.
trainX = np.array([])
trainY = np.array([])

for i in range(0, len(trainData)-pastSteps-futureSteps):
    trainX = np.append(trainX, trainData[i:i+pastSteps, 0])
    trainY = np.append(trainY, trainData[i+pastSteps:i+pastSteps+futureSteps, 0])

# Reshape data to 3D, first dimension is count of output data, second defines from how many values it will be
# predicted, and the last dimension is set to 1 (because we have only one type of data)
trainX = np.reshape(trainX, (int(len(trainX)/pastSteps), pastSteps, 1))
trainY = np.reshape(trainY, (int(len(trainY)/futureSteps), futureSteps))

# Build and compile the model, then save it to TICKER dir
model = Sequential()
model.add(LSTM(futureSteps, return_sequences=True, input_shape=(trainX.shape[1], 1)))
model.add(Dropout(0.1))
model.add(LSTM(futureSteps*2, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(futureSteps*3, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(futureSteps*2))
model.add(Dense(futureSteps))
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(trainX, trainY, validation_split=1, epochs=5)
model.save(os.path.join(folderPath, "model.h5"))
