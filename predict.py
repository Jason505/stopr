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
# timeShift = config.timeShift
timeShift = 300

folderPath = os.path.join(os.getenv("APPDATA"), "Stopr", config.ticker)
dataPath = os.path.join(folderPath, "data.csv")
scalerPath = os.path.join(folderPath, "scaler.dat")

predictLength = int(sum(1 for line in open(dataPath)))
df = pd.read_csv(dataPath)
data = df["Scaled"]
data = data[predictLength-pastSteps-timeShift:predictLength-timeShift]
predictData = data.to_numpy()

# Load model and predict next *futureSteps* values
model = load_model(os.path.join(folderPath, "model.h5"))
for i in range(0, futureSteps):
    toPredict = predictData[i:pastSteps+i]
    toPredict = np.reshape(toPredict, (1, pastSteps, 1))
    predictData = np.append(predictData, model.predict(toPredict))
predictData = predictData.reshape(-1, 1)

# Load scaler used in train part and descale predicted data
scaler = joblib.load(scalerPath)
predictedData = np.round(scaler.inverse_transform(predictData), decimals=3)
predictedData = predictedData[pastSteps:pastSteps+futureSteps, 0]

# From date of last non-generated value find out the date of first generated value, then make
# datasheet of dates for all generated data.
lastDate = df["Date"].to_numpy()
lastDate = lastDate[predictLength-timeShift-1]
lastDate = datetime.strptime(lastDate, "%Y-%m-%d")
lastDate = pd.DataFrame(rrule(DAILY, dtstart=lastDate, count=2, byweekday=(MO, TU, WE, TH, FR))).iloc[1, 0]
timestamp = rrule(DAILY, dtstart=lastDate, count=futureSteps, byweekday=(MO, TU, WE, TH, FR))

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
