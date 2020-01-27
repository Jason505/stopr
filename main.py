# Main file
from os import path

# Check if gui mode is present; if so, set bool value so other parts of script so they know it.
print("Do you want to run code in GUI mode? [Y/N]")
userInput = input()
if str.lower(userInput) == "n":
    guiMode = False
    print("Running code in command line . . .")
elif str.lower(userInput) == "y":
    if path.exists("gui.py"):
        guiMode = True
        print("Running code with GUI . . .")
    else:
        print("Files required for GUI not found.")
        print("Running code in command line . . .")
        guiMode = False
else:
    print("Invalid entry, assuming \"N\" as an answer . . .")
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
