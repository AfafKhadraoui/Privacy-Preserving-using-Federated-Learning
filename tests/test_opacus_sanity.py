import torch
import torch.nn as nn
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset

def test_opacus_sanity():
    print("=== [TEST] Opacus Sanity Check ===")
    
    # 1. Setup small model and data
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dataset = TensorDataset(torch.randn(16, 10), torch.zeros(16, dtype=torch.long))
    data_loader = DataLoader(dataset, batch_size=4)
    
    # 2. Wrap with Opacus
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
    )
    
    print("Opacus wrapping successful.")
    
    # 3. Step
    model.train()
    for x, y in data_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        break
        
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"One step successful. Epsilon: {epsilon:.4f}")
    print("[SUCCESS] Opacus is functional.")

if __name__ == "__main__":
    test_opacus_sanity()
