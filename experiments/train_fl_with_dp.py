"""
Purpose: The quick launcher script to train Version B.
It calls my run_fl manager but flips the switch to use_dp = True.
This version uses Opacus to wrap our optimizer in Differential Privacy noise, 
creating our secure model that foils the attack!
"""

import os
import sys

# Putting our root project folder on the python path!
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.federated import run_fl

if __name__ == "__main__":
    print("Launching training for Version B (WITH Opacus DP). Let's protect some faces! 🛡️")
    run_fl.main(use_dp=True)