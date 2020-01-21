# Download and prepair data table for predicting
import pandas as pd
import os
import yfinance as yf

forceUpdate = True
ticker = "MSFT"

# Let user choose ticker to predict, default is MSFT (Microsoft)
# TODO add check if the ticker exists
print("Type custom ticker or press Enter to continue with preselected ticker")
print("Default: " + ticker)
userInput = input()
if userInput != "":
    ticker = userInput
print("Using company " + ticker + " as a data source . . .\n")

# Paths for storing user data
appPath = os.path.join(os.getenv("APPDATA"), "Stopr")
filesPath = os.path.join(appPath, ticker)
dataPath = os.path.join(filesPath, "data.csv")

# Make paths if they do not exist and asks if the data should be updated
# TODO add automated procedure to find out if newer data is available
if not os.path.exists(appPath):
    os.mkdir(appPath)
if not os.path.exists(filesPath):
    os.mkdir(filesPath)
else:
    print("Do you want to force update data? [Y/N}")
    userInput = input()
    if str.lower(userInput) == "n":
        forceUpdate = False
    elif str.lower(userInput) != "y":
        print("Invalid entry, assuming \"Y\" as an answer . . .")

# Download, sort and save data if forceUpdate is enabled
if forceUpdate:
    data = yf.download(ticker, period="5Y", prepost=True).to_csv()
    with open(dataPath, "w+") as f:
        f.write(data)
    df = pd.read_csv(dataPath)
    df = df[["Date", "Open", "Low", "High", "Close"]]
    df.set_index("Date", inplace=True)
    df.sort_values(by=["Date"], ascending=True, inplace=True)
    df.to_csv(dataPath)

# Write important global variables to configfile
with open("config.py", "a") as f:
    try:
        config.ticker
    except NameError:
        f.write("ticker = " + "\"" + ticker + "\"\n")
