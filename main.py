# Main file
import os
import subprocess

# Create paths for files if they do not exist
# Also, if the app path does not exist, it is clear that the app is being run for the first time => install packages
appPath = os.path.join(os.getenv("APPDATA"), "Stopr")
if not os.path.exists(appPath):
    os.mkdir(appPath)
    subprocess.run("pip install -r packages.txt")

# Import subprograms
import download
import train
import predict
# import oldpredict
import plot
