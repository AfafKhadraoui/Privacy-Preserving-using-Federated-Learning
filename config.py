"""
Purpose: The main configuration file for the entire project. 
Every number and file path we use is stored right here. 

"""

import os

# Federated Learning Settings
NUM_ROUNDS    = 20    # Number of times the server and clients talk to each other
NUM_CLIENTS   = 5   
LOCAL_EPOCHS  = 5     # How many times each device trains on its own photos before sending weights
LEARNING_RATE = 1e-4

# Two-model strategy
# False = Version A (no DP, the vulnerable baseline)
# True  = Version B (with DP, the protected one that foils the attack)
USE_DP = False

# Differential Privacy (Opacus)
# These are only active when USE_DP is True
NOISE_MULTIPLIER = 1.1   # How much noise we add to gradients. Higher = more private but worse accuracy.
MAX_GRAD_NORM    = 1.0   # Clip the gradients so no single photo dominates the update
DELTA            = 1e-5

# Model Parameters
EMBEDDING_SIZE = 512
PRETRAINED     = "vggface2"

# Recognition
THRESHOLD = 0.6          # Distance threshold for face verification

# FL Server
SERVER_ADDRESS = "127.0.0.1:8080"  # Since we're simulating on localhost

# Paths
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))

# Data folders
DATA_DIR       = os.path.join(BASE_DIR, "data", "raw")
CROPPED_DIR    = os.path.join(BASE_DIR, "data", "cropped")          
CLIENTS_DIR    = os.path.join(BASE_DIR, "data", "clients")         
ENROLLMENT_DIR = os.path.join(BASE_DIR, "data", "enrollment")

# Results folders
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
MODELS_DIR     = os.path.join(BASE_DIR, "results", "models")
PLOTS_DIR      = os.path.join(BASE_DIR, "results", "plots")
METRICS_DIR    = os.path.join(BASE_DIR, "results", "metrics")

# Saved models
MODEL_CENTRALIZED = os.path.join(MODELS_DIR, "model_centralized.pth")
MODEL_NO_DP       = os.path.join(MODELS_DIR, "model_fl_no_dp.pth")    # The one we attack
MODEL_WITH_DP     = os.path.join(MODELS_DIR, "model_fl_with_dp.pth")  # The one that resists the attack!

# Attack
ATTACK_ITERATIONS = 1000
ATTACK_LR         = 0.01