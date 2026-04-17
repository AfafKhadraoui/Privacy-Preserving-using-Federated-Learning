"""
This acts like a conductor.
It uses Python 'multiprocessing' to fire up our server and all 5 clients
at EXACTLY the same time on my laptop. This tricks Flower into thinking they are run broadly on multiple phones
"""

import os
import sys
import multiprocessing
import time
import subprocess
import torch

# Always need to add the root project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

def run_server(use_dp):
    # Command to boot up my server script
    cmd = [sys.executable, os.path.join(config.BASE_DIR, "src", "federated", "server.py")]
    if use_dp:
        cmd.append("--use_dp")
    subprocess.run(cmd)

def run_client(client_id, use_dp):
    # Command to boot up a single client device
    cmd = [
        sys.executable, 
        os.path.join(config.BASE_DIR, "src", "federated", "client.py"),
        "--client_id", str(client_id)
    ]
    if use_dp:
        cmd.append("--use_dp")
    subprocess.run(cmd)

def main(use_dp=False):
    print("=" * 60)
    mode_text = "VERSION B (WITH DP)" if use_dp else "VERSION A (NO DP)"
    print(f"Starting My Federated Learning Simulation for {mode_text}")
    print("=" * 60)
    
    processes = []
    
    # 1. Start the server first
    server_process = multiprocessing.Process(target=run_server, args=(use_dp,))
    server_process.start()
    processes.append(server_process)
    
    # Give the server a few seconds to wake up and get ready on port 8080
    time.sleep(3)
    
    # 2. Discover all available client folders
    # We look for folders named 'client_xx' in the CLIENTS_DIR
    client_folders = sorted([
        d for d in os.listdir(config.CLIENTS_DIR) 
        if os.path.isdir(os.path.join(config.CLIENTS_DIR, d)) and d.startswith("client_")
    ])
    
    num_total_clients = len(client_folders)
    print(f"Found {num_total_clients} client folders in {config.CLIENTS_DIR}")
    
    if num_total_clients < config.MIN_CLIENTS:
        print(f"WARNING: Only found {num_total_clients} clients, but server needs {config.MIN_CLIENTS}.")
        print("Increasing the server's patience or add more data folders!")

    # --- SANITY CHECKS TO PREVENT PC CRASHING ---
    # 1. Limit CPU threads per process so multiprocessing doesn't freeze the system
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"System memory available: {mem.available / (1024**3):.2f} GB")
        if mem.available < 4 * 1024**3:
            print("WARNING: You have less than 4GB of free RAM! The simulation might crash.")
    except ImportError:
        pass
    # --------------------------------------------
    
    # 3. Wake up all the client devices staggeredly
    for i in range(num_total_clients):
        client_proc = multiprocessing.Process(target=run_client, args=(i, use_dp))
        client_proc.start()
        processes.append(client_proc)
        # Sleep to staggered memory loading and prevent immediate OOM killer
        print(f"Started client_0{i}. Waiting 3 seconds before starting next one to avoid RAM spike...")
        time.sleep(3)
        
    # 4. Wait for the round of schooling to finish!
    for p in processes:
        p.join()
        
    print("All processes have finished.")
    
    # After everyone stops, the last state of the clients has the newest weights.
    
    print("Simulation complete! Check the results folder for the metrics.")

if __name__ == "__main__":
    # If run directly without the wrapper, just default to Version A
    main(use_dp=False)