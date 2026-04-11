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
    
    # 2. Wake up all the client devices
    for i in range(config.NUM_CLIENTS):
        client_proc = multiprocessing.Process(target=run_client, args=(i, use_dp))
        client_proc.start()
        processes.append(client_proc)
        
    # 3. Wait for the round of schooling to finish!
    for p in processes:
        p.join()
        
    print("All processes have finished.")
    
    # After everyone stops, the last state of the clients has the newest weights.
    
    print("Simulation complete! Check the results folder for the metrics.")

if __name__ == "__main__":
    # If run directly without the wrapper, just default to Version A
    main(use_dp=False)