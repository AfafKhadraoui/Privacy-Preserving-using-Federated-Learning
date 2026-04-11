"""
Purpose: My quick little script to simulate the FL devices! 
It reads all the  cropped faces and splits them 
up into 5 "client" folders. This fakes the fact that we're on one laptop instead of 5 real smartphones.
"""


import os
import shutil
import sys

# Hack so we can easily import config no matter where we run this script from
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

def split_into_clients():
    print(f"Reading from Abderrahim's output folder: {config.CROPPED_DIR}...")
    
    # Check if the cropped data even exists
    if not os.path.exists(config.CROPPED_DIR):
        print("The cropped directory is missing.")
        return
        
    people = [d for d in os.listdir(config.CROPPED_DIR) if os.path.isdir(os.path.join(config.CROPPED_DIR, d))]
    
    if not people:
        print("Empty folder! We really need those photos before we can partition anything.")
        return

    # Let's sort to keep things deterministic
    people.sort()
    
    # If I already ran this before, just blow away the old clients directory to stay fresh
    if os.path.exists(config.CLIENTS_DIR):
        shutil.rmtree(config.CLIENTS_DIR)
        
    os.makedirs(config.CLIENTS_DIR, exist_ok=True)
    
    # We assign exactly one person's folder to one client device
    for idx, person in enumerate(people):
        client_id_str = f"client_{idx:02d}"
        client_path = os.path.join(config.CLIENTS_DIR, client_id_str)
        os.makedirs(client_path, exist_ok=True)
        
        # Copying the folder inside
        dest_person_path = os.path.join(client_path, person)
        src_person_path = os.path.join(config.CROPPED_DIR, person)
        
        shutil.copytree(src_person_path, dest_person_path)
        print(f"Partitioned: {person}'s sweet photos went to {client_id_str}")

if __name__ == "__main__":
    split_into_clients()
    print("Done partitioning data! We are good to go for Federated Learning.")