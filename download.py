# Download and prepare data table for predicting
import pandas as pd
import os
import yfinance as yf

forceUpdate = True
ticker = "MSFT"

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


# Paths for storing user data
appPath = os.path.join(os.getenv("APPDATA"), "stopr")
filesPath = os.path.join(appPath, ticker)
dataPath = os.path.join(filesPath, "data.csv")

# Make paths if they do not exist and asks if the data should be updated
if not os.path.exists(filesPath):
    os.mkdir(filesPath)

# Download, sort and save data if forceUpdate is enabled
print("Update of ticker data in progress . . .")
data = yf.download(ticker, period="5Y", prepost=True).to_csv()
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
    except NameError:
        f.write("ticker = " + "\"" + ticker + "\"\n")
