# Take prepared data and train model on it
from datetime import datetime
from dateutil import *
from dateutil.rrule import *
from functions import *
import joblib
from keras.models import load_model
from keras.losses import mean_absolute_percentage_error
import numpy as np
import os
import pandas as pd
from time import time_ns

reload_config()
from config import *
# Initialization of essential paths
folderPath = os.path.join(os.getenv("APPDATA"), "stopr", ticker)
dataPath = os.path.join(folderPath, "data.csv")
scalerPath = os.path.join(folderPath, "scaler.dat")

predictLength = int(sum(1 for line in open(dataPath)))
df = pd.read_csv(dataPath)
data = df["Scaled"]
trueData = df["Close"]
predictData = data[predictLength-pastSteps-timeShift:predictLength-timeShift]
trueData = trueData[predictLength-timeShift:predictLength+-timeShift+futureSteps]
predictData = predictData.to_numpy()

# Load model and predict next *futureSteps* values
startT = time_ns()
try:
    model = load_model(os.path.join(folderPath, "models", "model.h5"))
except:
    pass

print("Predicting in progress . . .")
predictedData = []
if modelType == 1:
    predictData = np.reshape(predictData, (1, pastSteps, 1))
    predictedData = np.append(predictedData, model.predict(predictData))
elif modelType == 2:
    for i in range(0, futureSteps):
        toPredict = predictData[i:pastSteps + i]
        toPredict = np.reshape(toPredict, (1, pastSteps, 1))
        predictData = np.append(predictData, model.predict(toPredict))
    predictedData = predictData[pastSteps:pastSteps + futureSteps]
elif modelType == 3:
    predictData = data[predictLength-pastSteps-timeShift:predictLength-timeShift+futureSteps]
    predictData = predictData.to_numpy()
    for i in range(0, futureSteps):
        toPredict = predictData[i:pastSteps + i]
        toPredict = np.reshape(toPredict, (1, pastSteps, 1))
        predictedData = np.append(predictedData, model.predict(toPredict))
else:
    for i in range(0, futureSteps):
        print("\nPredicting " + str(i+1) + ". step . . .")
        model = load_model(os.path.join(folderPath, "models", "model_" + str(i) + ".h5"))
        toPredict = predictData[:pastSteps]
        toPredict = np.reshape(toPredict, (1, pastSteps, 1))
        predictedData = np.append(predictedData, model.predict(toPredict))
        del model
endT = time_ns()
deltaT = str(round(float(endT - startT)/1000000000, 3))
print("Elapsed time: " + deltaT + " s")

# Load scaler used in train part and descale predicted data
predictedData = predictedData.reshape(-1, 1)
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

# Calculate the MSE
MAPE = mean_absolute_percentage_error(trueData, predictedData).numpy()
MAPE = np.average(MAPE)
MAPE = round(MAPE, 2)
print("MAPE: " + str(MAPE) + "%")