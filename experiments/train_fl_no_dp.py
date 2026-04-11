"""
Purpose: The quick launcher script to train Version A.
It just calls my run_fl manager with the Differential Privacy switch set to False.
This version of the model is intentionally vulnerable so we can attack it and show 
how bad things could be without  DP code 
"""

import os
import sys

# Putting our root project folder on the python path!
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.federated import run_fl

if __name__ == "__main__":
    print(" Launching training for Version A (No DP). Let's build a vulnerable model to attack! 🚀")
    run_fl.main(use_dp=False)