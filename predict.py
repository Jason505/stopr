# Take prepared data and train model on it
from datetime import datetime
from dateutil import *
from dateutil.rrule import *
import joblib
from keras.models import load_model
import numpy as np
import os
import pandas as pd

# Reload config for eventual changes
import importlib
import config
importlib.reload(config)

futureSteps = config.futureSteps
pastSteps = config.pastSteps
timeShift = config.timeShift

folderPath = os.path.join(os.getenv("APPDATA"), "stopr", config.ticker)
dataPath = os.path.join(folderPath, "data.csv")
scalerPath = os.path.join(folderPath, "scaler.dat")

predictLength = int(sum(1 for line in open(dataPath)))
df = pd.read_csv(dataPath)
data = df["Scaled"]
data = data[predictLength-pastSteps-timeShift:predictLength-timeShift]
predictData = data.to_numpy()

# Load model and predict next *futureSteps* values
model = load_model(os.path.join(folderPath, "model.h5"))
predictData = np.reshape(predictData, (1, pastSteps, 1))
predictedData = np.append(predictData[0, pastSteps-1:pastSteps, 0], model.predict(predictData))
predictedData = predictedData.reshape(-1, 1)

# Load scaler used in train part and descale predicted data
scaler = joblib.load(scalerPath)
predictedData = np.round(scaler.inverse_transform(predictedData), decimals=2)

# From date of last non-generated value find out the date of first generated value, then make
# datasheet of dates for all generated data.
lastDate = df["Date"].to_numpy()
lastDate = lastDate[predictLength-timeShift-1]
lastDate = datetime.strptime(lastDate, "%Y-%m-%d")
timestamp = rrule(DAILY, dtstart=lastDate, count=futureSteps+1, byweekday=(MO, TU, WE, TH, FR))

# Concat the dates with values and save it to predict.csv
timestamp = pd.DataFrame(timestamp).reset_index(drop=True)
saveData = pd.concat([timestamp, pd.DataFrame(predictedData)], axis=1, ignore_index=True)
saveData.columns = ["Date", "Value"]
saveData.to_csv(os.path.join(folderPath, "predict.csv"), index=False)

# TODO Fix RMSE calculation & add other error calculations
# Calculate the RMSE
# rmse = np.sqrt(((predictedData - predictY) ** 2).mean())
# print("Calculated RMSE:")
# print(rmse)
