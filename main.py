from logic.server.server import start_server
from logic.ai.stock_model import create_stock_model
from logic.ai.fit import Fit
from termcolor import cprint
import pyfiglet
import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def send_help():
    #Sends user the help-commands list 

    cprint("Check all the available commands", color="green")
    cprint("""
        --fit: runs the training process\n
        --prod: runs the production mode
        """, color="red")

def send_prod():
    #Sends user the production mode ...

    cprint(pyfiglet.figlet_format("AI is ready!", font="bubble"), color="green")
    start_server()

def send_fit():
    #Sends user the fitting mode ...

    cprint(pyfiglet.figlet_format("AI is fitting!", font="bubble"), color="yellow")
    Fit(create_stock_model())

def handle_flags():
    #Handles available commands ...

    if sys.argv[-1] == "--fit":
        send_fit()
    elif sys.argv[-1] == "--prod":
        send_prod()
    else:
        send_help()

if __name__ == "__main__":
    handle_flags()

    
