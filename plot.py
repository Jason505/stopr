# Plot the historic & predicted data
from config import *
from functions import *
import os
import matplotlib.pyplot as plt
import pandas as pd

reload_config()
from config import *
# For future support
pd.plotting.register_matplotlib_converters()

# Initialization of essential paths
folderPath = os.path.join(os.getenv("APPDATA"), "stopr", ticker)
dataPath = os.path.join(folderPath, "data.csv")
predictPath = os.path.join(folderPath, "predict.csv")

# Getting historic data from "data.csv"
df = pd.read_csv(dataPath)
data = df["Close"].to_numpy()
data = data[len(data)-timeShift-futureSteps:len(data)-timeShift+futureSteps*2]
dataDate = df["Date"].to_numpy()
dataDate = dataDate[len(dataDate)-timeShift-futureSteps:len(dataDate)-timeShift+futureSteps*2]
dataDate = dataDate.astype("datetime64")

# Getting prediction data from "predict.csv"
df = pd.read_csv(predictPath)
predict = df["Value"].to_numpy()
predictDate = df["Date"].to_numpy()
predictDate = predictDate.astype("datetime64")

# Plot the graph with used style
# plt.style.use("dark.mplstyle")
plt.margins(0, 0, tight=True)
plt.rcParams.update({"font.size": 16})
plt.title("Close price of ticker " + ticker)
plt.xlabel("Datestamp", fontsize=14)
plt.ylabel("Close price (USD)", fontsize=14)
historic = plt.plot(dataDate, data, label="Historic data")
predicted = plt.plot(predictDate, predict, label="Predicted data")
plt.tight_layout()
plt.plot()
plt.legend()
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.show()
