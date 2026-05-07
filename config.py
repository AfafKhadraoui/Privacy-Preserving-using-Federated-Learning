"""
Purpose: The main configuration file for the entire project. 
Every number and file path we use is stored right here. 

"""

import os

# Federated Learning Settings
NUM_ROUNDS    = 20    # Number of times the server and clients talk to each other
MIN_CLIENTS   = 5   
LOCAL_EPOCHS  = 5     # How many times each device trains on its own photos before sending weights
LEARNING_RATE = 1e-4

# Two-model strategy
# False = Version A (no DP, the vulnerable baseline)
# True  = Version B (with DP, the protected one that foils the attack)
# Differential Privacy Settings
# ----------------------------
# DP_MODE options:
#   "opacus"      - Original Opacus DP-SGD (sample-level, memory intensive)
#   "manual_sgd"  - Manual gradient clipping + noise (sample-level substitute)
#   "embedding"   - Local DP on embeddings (add noise to 512-dim vectors)
#   "client"      - Client-level DP (add noise to weight updates)
#   "none"        - No DP (active when USE_DP is False)
DP_MODE = "embedding"  # Default to original implementation

# These are only active when USE_DP is True
NOISE_MULTIPLIER = 1.1   # How much noise we add to gradients. Higher = more private but worse accuracy.
MAX_GRAD_NORM    = 1.0   # Clip the gradients so no single photo dominates the update
DELTA            = 1e-5
EPSILON_MAX      = 5.0   # Hard privacy budget cap (training should stop if exceeded)
DP_RANDOM_SEED   = 42    # Reproducibility seed for DP experiments
PROTOCOL_VERSION = "v1" # Version used in signed metadata / protocol integrity checks
ENFORCE_SIGNATURES = True
CRYPTO_DEBUG_LOGS = True
CRYPTO_DEBUG_SHOW_FULL_KEYS = True

# Model Parameters
EMBEDDING_SIZE = 512
PRETRAINED     = "vggface2"

# Recognition
THRESHOLD = 0.6          # Distance threshold for face verification

# Registration
NUM_REGISTRATION_IMAGES = 3
DEFAULT_REGISTRATION_IMAGE_INSTRUCTIONS = (
    "Look straight",
    "Turn left",
    "Turn right",
)
if NUM_REGISTRATION_IMAGES < 1:
    raise ValueError("NUM_REGISTRATION_IMAGES must be at least 1")
REGISTRATION_IMAGE_INSTRUCTIONS = tuple(
    DEFAULT_REGISTRATION_IMAGE_INSTRUCTIONS[idx]
    if idx < len(DEFAULT_REGISTRATION_IMAGE_INSTRUCTIONS)
    else f"Capture image {idx + 1}"
    for idx in range(NUM_REGISTRATION_IMAGES)
)

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
KEYS_DIR       = os.path.join(BASE_DIR, "data", "keys")

# Saved models
MODEL_CENTRALIZED = os.path.join(MODELS_DIR, "model_centralized.pth")
MODEL_NO_DP       = os.path.join(MODELS_DIR, "model_fl_no_dp.pth")    # The one we attack
MODEL_WITH_DP     = os.path.join(MODELS_DIR, "model_fl_with_dp.pth")  # The one that resists the attack!

# Attack (model inversion)
# Pixel-space inversion: optimizes image pixels directly to match target embedding.
ATTACK_ITERATIONS = 200
ATTACK_LR = 0.02
# After preprocessing, which client folders to evaluate (different identities / test photos).
ATTACK_EVAL_CLIENT_IDS = ("client_00", "client_01")

# StyleGAN inversion disabled (removed to simplify attack to pixel-space only)
STYLEGAN_NETWORK_PKL = None
STYLEGAN_REPO_DIR = None
STYLEGAN_REPO_URL = None

STYLEGAN_IDENTITY_W    = 1.0
STYLEGAN_PERCEPTUAL_W  = 0.1
STYLEGAN_LATENT_REG_W  = 0.001

