# Main file
from os import path

# Check if gui mode is present; if so, set bool value so other parts of script so they know it.
# TODO add manual asking if user wants GUI or cmd version
if path.exists("gui.py"):
    guiMode = True
else:
    guiMode = False

# Write variables to config.py
with open("config.py", "w+") as f:
    f.write("guiMode = " + str(guiMode))

# Import subprograms
# TODO add library checking and automatic installation of them
import download
import train
import predict
# import oldpredict
import plot
