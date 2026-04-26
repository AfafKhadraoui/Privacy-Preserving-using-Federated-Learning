import threading
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import src.federated.run_fl as run_fl
import config as proj_cfg

def trigger_federated_learning(use_dp: bool = False):
    """
    Triggers the FL pipeline in a background thread to avoid blocking the API.
    """
    def run_fl_bg():
        print("[API Server] Starting Federated Learning pipeline...")
        # Override the MIN_CLIENTS if we don't have enough registered yet to allow testing
        client_folders = [d for d in os.listdir(proj_cfg.CLIENTS_DIR) if d.startswith("client_")]
        
        # Keep original config safe
        original_min = proj_cfg.MIN_CLIENTS
        if len(client_folders) < proj_cfg.MIN_CLIENTS:
            print(f"[API Server] Adjusting MIN_CLIENTS from {proj_cfg.MIN_CLIENTS} to {len(client_folders)} for testing.")
            proj_cfg.MIN_CLIENTS = max(1, len(client_folders))
            
        try:
            run_fl.main(use_dp=use_dp)
        finally:
            # Restore original config
            proj_cfg.MIN_CLIENTS = original_min
            print("[API Server] Federated Learning pipeline completed.")

    thread = threading.Thread(target=run_fl_bg)
    thread.start()
    return True
