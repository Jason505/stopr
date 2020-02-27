# Define functions for writing variables
def writevar(name):  # For user input
    command = "config." + name
    try:
        exec(command)
    except (NameError, AttributeError):
        var_written = False
        while not var_written:
            print("Enter value of \'" + name + "\":")
            try:
                user_input = int(input())
                var_written = True
            except ValueError:
                var_written = False
                print("ERROR! Invalid entry . . .")
        with open("config.py", "a") as f:
            f.write("\n" + name + " = " + str(user_input))


def writevar(name, value):  # For predefined values
    command = "config." + name
    try:
        exec(command)
    except (NameError, AttributeError):
        with open("config.py", "a") as f:
            f.write("\n" + name + " = " + str(value))


# Reloading config file
def reload_config():
    import importlib
    import config
    importlib.reload(config)
