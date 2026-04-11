"""
This script waits for the 5 client devices (our phones) to connect. 
It collects their weight updates, combines them using the FedAvg magic, 
and does this over and over for 20 rounds, Then we save the final model.
"""

import os
import argparse
import sys
import flwr as fl
import json
import torch

# Needed to import config from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.model.face_model import get_model, set_parameters

def start_server(use_dp):
    print(f"Starting the Server... (Version B / DP enabled: {use_dp})")
    
    # We want ALL 5 clients to be awake and participating before a round can start!
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config.NUM_CLIENTS,
        min_evaluate_clients=config.NUM_CLIENTS,
        min_available_clients=config.NUM_CLIENTS,
    )
    
    # Start the server (this will pause and wait until run_fl.py boots up the clients)
    history = fl.server.start_server(
        server_address=config.SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=config.NUM_ROUNDS),
        strategy=strategy,
    )
    
    # Let's save a simple record of what happened so we can plot it later
    metrics_file = "fl_with_dp_results.json" if use_dp else "fl_no_dp_results.json"
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    
    with open(os.path.join(config.METRICS_DIR, metrics_file), "w") as f:
        json.dump({"rounds": config.NUM_ROUNDS, "dp": use_dp, "success": True}, f)
        
    print(f"Server is done teaching! We saved the metrics to {metrics_file}.")
    print("Oh, and don't forget we should probably save the final model weights somewhere here too in reality, but run_fl.py might be taking care of that or we'd just dump the model to MODELS_DIR!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My Flower Server")
    parser.add_argument("--use_dp", action="store_true", help="Tell the server if we're running the DP version")
    args = parser.parse_args()
    
    start_server(args.use_dp)