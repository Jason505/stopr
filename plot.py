# Plot the historic & predicted data
import config
import os
import matplotlib.pyplot as plt
import pandas as pd

# For future support
pd.plotting.register_matplotlib_converters()

# Initialization of essential paths
folderPath = os.path.join(os.getenv("APPDATA"), "Stopr", config.ticker)
dataPath = os.path.join(folderPath, "data.csv")
predictPath = os.path.join(folderPath, "predict.csv")

# Getting historic data from "data.csv"
df = pd.read_csv(dataPath)
data = df["Close"].to_numpy()
dataDate = df["Date"].to_numpy()
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
