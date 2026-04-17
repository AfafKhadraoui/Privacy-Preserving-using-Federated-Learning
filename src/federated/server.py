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
from src.model.face_model import get_model, set_parameters, save_model

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, use_dp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_dp = use_dp

    def aggregate_fit(self, server_round, results, failures):
        # 1. Ask the generic FedAvg class to actually calculate the weighted average!
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # 2. If we are on the very last round and aggregation was successful -> SAVE THE MODEL!
        if aggregated_parameters is not None and server_round == config.NUM_ROUNDS:
            print(f"\n---> [Round {server_round}] Simulation finished! Extracting final weights...")
            
            # Convert Flower's byte parameters back to numpy arrays
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            # Wake up Kosai's model and inject the final averaged weights
            model = get_model()
            set_parameters(model, ndarrays)
            
            # Where are we saving this?
            save_path = config.MODEL_WITH_DP if self.use_dp else config.MODEL_NO_DP
            save_model(model, save_path)
            
            print(f"---> SUCCESS: Final global model permanently saved to {save_path}!\n")
            
        return aggregated_parameters, aggregated_metrics

def weighted_average(metrics: list) -> dict:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def start_server(use_dp):
    print(f"Starting the Server... (Version B / DP enabled: {use_dp})")
    
    from src.privacy.secure_agg import get_secagg_strategy
    
    # We use our custom strategy so we can intercept and save the model at the end
    # We wrap it in SecAgg so that the connection is secure!
    strategy = get_secagg_strategy(
        base_strategy_class=SaveModelStrategy,
        use_dp=use_dp,
        fraction_fit=1.0,          # Sample 100% of available clients
        fraction_evaluate=1.0,
        min_fit_clients=config.MIN_CLIENTS,
        min_evaluate_clients=config.MIN_CLIENTS,
        min_available_clients=config.MIN_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My Flower Server")
    parser.add_argument("--use_dp", action="store_true", help="Tell the server if we're running the DP version")
    args = parser.parse_args()
    
    start_server(args.use_dp)