# Deprecated script for training data; not updated for a while
from dateutil import *
from dateutil.rrule import *
from keras.models import load_model
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Reload config for eventual changes
import importlib
import config
importlib.reload(config)

futureSteps = config.futureSteps
pastSteps = config.pastSteps

folderPath = os.path.join(os.getenv("APPDATA"), "Stopr", config.ticker)
dataPath = os.path.join(folderPath, "data.csv")

predictLength = int(sum(1 for line in open(dataPath)))
df = pd.read_csv(dataPath)
data = df["Close"]
numpyData = data.to_numpy()

# Take second group for predicting and scale it
lineCount = sum(1 for line in open(dataPath))
entryData = numpyData[predictLength - 2*pastSteps-1:]

scaler = MinMaxScaler(feature_range=(0, 1))
entryDataScaled = scaler.fit_transform(entryData.reshape(-1, 1))

# Again, split the group to X and Y for same reason as in train code
# Here, the Y part will be the predicted data
predictX = []

for i in range(pastSteps, len(entryDataScaled)):
    predictX.append(entryDataScaled[i - pastSteps:i, 0])
predictX = np.array(predictX)

# Reshape the data to 3D
predictX = np.reshape(predictX, (predictX.shape[0], predictX.shape[1], 1))

# Import model, predict data and de-scale them
model = load_model(os.path.join(folderPath, "model.h5"))
predictedData = model.predict(predictX)
predictedData = np.round(scaler.inverse_transform(predictedData), decimals=3)

# Add timestamp and save predicted values to predicted.csv
today = utils.today()
today = today+relativedelta.relativedelta(days=-120)
timestamp = rrule(DAILY, dtstart=today, count=120, byweekday=(MO, TU, WE, TH, FR))
timestamp = pd.DataFrame(timestamp).reset_index(drop=True)
saveData = pd.concat([timestamp, pd.DataFrame(predictedData)], axis=1, ignore_index=True)
saveData.columns = ["Date", "Value"]
saveData.to_csv(os.path.join(folderPath, "predict.csv"), index=False)

# Calculate the RMSE
# rmse = np.sqrt(((predictedData - predictY) ** 2).mean())
# print("Calculated RMSE:")
# print(rmse)
