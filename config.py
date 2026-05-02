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
USE_DP = False

# Differential Privacy (Opacus)
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
# Pixel-space inversion needs thousands of steps; StyleGAN latent search also needs 1500+.
ATTACK_ITERATIONS = 2500
ATTACK_LR = 0.02
# After preprocessing, which client folders to evaluate (different identities / test photos).
ATTACK_EVAL_CLIENT_IDS = ("client_00", "client_01")

# StyleGAN inversion (recommended for face-like inversions). dnnlib can load a URL or local .pkl.
# Default: official StyleGAN2-ADA FFHQ checkpoint (~364 MB, cached after first download).
# Override with a local path, e.g. STYLEGAN_NETWORK_PKL=D:\\models\\ffhq.pkl
# Disable StyleGAN (pixel-space only): set STYLEGAN_NETWORK_PKL to empty, none, 0, or false.
_stylegan_pkl_env = os.environ.get("STYLEGAN_NETWORK_PKL")
if _stylegan_pkl_env is None:
    STYLEGAN_NETWORK_PKL = (
        "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
    )
else:
    _s = _stylegan_pkl_env.strip()
    STYLEGAN_NETWORK_PKL = "" if _s.lower() in ("", "none", "0", "false") else _s
STYLEGAN_REPO_DIR      = os.environ.get("STYLEGAN_REPO_DIR", "")
STYLEGAN_REPO_URL      = os.environ.get(
	"STYLEGAN_REPO_URL",
	"https://github.com/NVlabs/stylegan2-ada-pytorch/archive/refs/heads/main.zip",
)
STYLEGAN_IDENTITY_W = 1.0
STYLEGAN_LATENT_REG_W = 0.001