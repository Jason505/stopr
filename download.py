# Download and prepare data table for predicting
from functions import *
import pandas as pd
import os
from shutil import rmtree
import yfinance as yf

forceUpdate = True
ticker = "MSFT"

# Should be the code run in developer mode (no asking for variables)?
print("Run the code in developer mode? [Y/N]")
print("(The application won't ask for any variables except ticker)")
userInput = input().upper()
if userInput == "Y":
    devMode = True
elif userInput == "N":
    devMode = False
else:
    print("Invalid entry! Assuming Y as an input . . .") # TODO change this for final release
    devMode = True

# Let user choose ticker to predict, default is MSFT (Microsoft)
invalidTicker = True
while invalidTicker:
    print("Type custom ticker or press Enter to continue with preselected ticker . . .")
    print("Default: " + ticker)
    userInput = input().upper()
    if userInput != "":
        checkData = yf.download(userInput, period="1d")
        if not checkData.empty:
            ticker = userInput
            invalidTicker = False
        else:
            print("ERROR! Non-existing ticker entered . . .")
    else:
        invalidTicker = False
print("Using company " + ticker + " as a data source . . .\n")

print("Which technique for predicting do you want to use?")
print("Default: 1")
print("1 - One network for whole prediction")
print("2 - One step ahead, and for next day use predicted value")
print("3 - One step ahead, and for next day use historic value")
print("4 - One neural network for each day [RAM demanding]")
try:
    modelType = int(input())
except (NameError, ValueError):
    print("Invalid value! Assuming option 1 as an answer . . .")
    modelType = 1

# Paths for storing user data
appPath = os.path.join(os.getenv("APPDATA"), "stopr")
filesPath = os.path.join(appPath, ticker)
dataPath = os.path.join(filesPath, "data.csv")

# Make paths if they do not exist and asks if the data should be updated
if not os.path.exists(filesPath):
    os.mkdir(filesPath)
rmtree(os.path.join(filesPath, "models"))
os.mkdir(os.path.join(filesPath, "models"))

# Download, sort and save data if forceUpdate is enabled
print("Update of ticker data in progress . . .")
data = yf.download(ticker, period="max", prepost=True).to_csv()
with open(dataPath, "w+") as f:
    f.write(data)
df = pd.read_csv(dataPath)
df = df[["Date", "Open", "Low", "High", "Close"]]
df.set_index("Date", inplace=True)
df.sort_values(by=["Date"], ascending=True, inplace=True)
df.to_csv(dataPath)

# Write important global variables to configfile
with open("config.py", "w+") as f:
    try:
        config.ticker
    except (NameError, AttributeError):
        f.write("ticker = " + "\"" + ticker + "\"\n")
    try:
        config.devMode
    except (NameError, AttributeError):
        f.write("devMode = " + str(devMode))
writevar("modelType", modelType)
