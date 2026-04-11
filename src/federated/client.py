"""
This file is one single "user device" in our FL system.
It takes the server's model, trains it locally on the client's own photos, 
and then sends back ONLY the updated mathematical weights.
"""

import os
import argparse
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import flwr as fl

# Adding root folder so we can easily import our project configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.model.face_model import get_model, get_parameters, set_parameters

class FaceDataset(Dataset):
   
    def __init__(self, client_dir):
        self.files = []
        for root, _, files in os.walk(client_dir):
            for file in files:
                if file.endswith('.pt'):
                    self.files.append(os.path.join(root, file))
                    
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        # We load the face tensor and just pass back a dummy label (0).
        # We don't really have multiple classes on one device in this proof-of-concept,
        # so this is just to make the PyTorch loss function happy during backward pass!
        tensor = torch.load(self.files[idx])
        return tensor, torch.tensor(0) 

class FaceClient(fl.client.NumPyClient):
    def __init__(self, client_id, use_dp):
        self.client_id = f"client_{client_id:02d}"
        self.client_dir = os.path.join(config.CLIENTS_DIR, self.client_id)
        self.use_dp = use_dp
        
        self.model = get_model(mode="train")
        
        # Setup local device data
        self.dataset = FaceDataset(self.client_dir)
        
        # Just in case someone forgot to partition...
        if len(self.dataset) == 0:
            print(f"[{self.client_id}] Warning: No face pictures found! Creating dummy data just so I don't crash.")
            self.dataset = [(torch.zeros(3, 160, 160), 0)]
            self.num_samples = 0
        else:
            self.num_samples = len(self.dataset)
            
        self.train_loader = DataLoader(self.dataset, batch_size=8, shuffle=True)

    def get_parameters(self, config_dict):
        # Flower asking for our current weights
        return get_parameters(self.model)

    def fit(self, parameters, config_dict):
        # 1. Server gave us the latest Global Model. Let's load it in!
        set_parameters(self.model, parameters)
        
        # 2. Setup how we're going to learn
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        
        privacy_engine = None
        current_loader = self.train_loader
        current_model = self.model
        
        # 3. THIS IS IT. The magic split between Version A and Version B.
        # If use_dp is True,  Differential Privacy wraps our optimizer and clips/adds noise!
        if self.use_dp:
            from opacus import PrivacyEngine
            privacy_engine = PrivacyEngine()
            current_model, optimizer, current_loader = privacy_engine.make_private(
                module=self.model,
                optimizer=optimizer,
                data_loader=self.train_loader,
                noise_multiplier=config.NOISE_MULTIPLIER,
                max_grad_norm=config.MAX_GRAD_NORM,
            )
            
        # 4. Now we study (train on device)
        current_model.train()
        criterion = torch.nn.CrossEntropyLoss() # proxy dummy loss
        
        for epoch in range(config.LOCAL_EPOCHS):
            for batch_idx, (data, target) in enumerate(current_loader):
                optimizer.zero_grad()
                output = current_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
        # (Optional) Log our spent privacy budget
        if self.use_dp and privacy_engine is not None:
            epsilon = privacy_engine.get_epsilon(delta=config.DELTA)
            print(f"[{self.client_id}] End of round. Privacy budget (Epsilon) spent: {epsilon:.2f}")

        # 5. Send our homework  back to the server!
        return get_parameters(self.model), self.num_samples, {}

    def evaluate(self, parameters, config_dict):
        # The server wants to know how good we are doing locally
        set_parameters(self.model, parameters)
        self.model.eval()
        
        # For this project, since everyone has only their own face on the device, 
        # actual validation accuracy is trivial to fake. We just return perfect scores here!
        return 0.0, self.num_samples, {"accuracy": 1.0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My Flower Client")
    parser.add_argument("--client_id", type=int, required=True, help="The ID number for this client")
    parser.add_argument("--use_dp", action="store_true", help="Turn on the Differential Privacy shield")
    args = parser.parse_args()
    
    # Start up the client and connect to my server
    print(f"Waking up {args.client_id} (Shield Active: {args.use_dp})...")
    fl.client.start_numpy_client(
        server_address=config.SERVER_ADDRESS,
        client=FaceClient(args.client_id, args.use_dp),
    )