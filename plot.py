# Plot the historic & predicted data
import config
import os
import matplotlib.pyplot as plt
import pandas as pd

# Reload config for eventual changes
import importlib
import config
importlib.reload(config)

futureSteps = config.futureSteps
pastSteps = config.pastSteps
timeShift = config.timeShift

# For future support
pd.plotting.register_matplotlib_converters()

# Initialization of essential paths
folderPath = os.path.join(os.getenv("APPDATA"), "stopr", config.ticker)
dataPath = os.path.join(folderPath, "data.csv")
predictPath = os.path.join(folderPath, "predict.csv")

# Getting historic data from "data.csv"
df = pd.read_csv(dataPath)
data = df["Close"].to_numpy()
data = data[len(data)-pastSteps-futureSteps:len(data)-futureSteps]
dataDate = df["Date"].to_numpy()
dataDate = dataDate[len(dataDate)-pastSteps-futureSteps:len(dataDate)-futureSteps]
dataDate = dataDate.astype("datetime64")

# Getting prediction data from "predict.csv"
df = pd.read_csv(predictPath)
predict = df["Value"].to_numpy()
predictDate = df["Date"].to_numpy()
predictDate = predictDate.astype("datetime64")

# Plot the graph with used
plt.style.use("dark.mplstyle")
plt.title("Close price of ticker " + config.ticker)
plt.xlabel("Datestamp")
plt.ylabel("Close price (USD)")
historic = plt.plot(dataDate, data, label="Historic data")
predicted = plt.plot(predictDate, predict, label="Predicted data")
plt.margins(0)
plt.plot()
plt.legend()
plt.show()
