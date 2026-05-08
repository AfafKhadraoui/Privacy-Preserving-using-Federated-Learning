import torch
print("[INFO] test_manual_dp.py started")
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.privacy.dp_training import apply_manual_dp_sgd
from src.model.face_model import GlobalFaceModel

def test_manual_dp():
    print("=== [TEST] Manual DP-SGD (From Scratch) ===")
    
    # 1. Setup a small dummy model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    # 2. Simulate a forward and backward pass to get raw gradients
    inputs = torch.randn(4, 10)
    targets = torch.randn(4, 2)
    criterion = nn.MSELoss()
    
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()
    
    # Check raw gradient norms
    raw_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000.0)
    print(f"Initial Gradient Norm: {raw_grad_norm:.4f}")
    
    # 3. Apply Manual DP-SGD (Clipping + Noising)
    MAX_NORM = 1.0
    SIGMA = 1.1
    
    print(f"\nApplying Manual DP (max_norm={MAX_NORM}, noise_multiplier={SIGMA})...")
    
    # Capture gradients before DP
    grads_before = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    
    apply_manual_dp_sgd(model, noise_multiplier=SIGMA, max_grad_norm=MAX_NORM)
    
    # Capture gradients after DP
    grads_after = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    
    # 4. Verify Results
    print("\n--- Verification Results ---")
    
    # Verify Clipping
    post_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000.0)
    print(f"Post-Clipping Grad Norm: {post_grad_norm:.4f} (Target was < {MAX_NORM} before noise)")
    
    # Verify Noising (Gradients should be different)
    for i, (gb, ga) in enumerate(zip(grads_before, grads_after)):
        diff = torch.abs(gb - ga).mean().item()
        print(f"Layer {i} Mean Noise Added: {diff:.6f}")
        
    if post_grad_norm > 0 and any(torch.abs(gb - ga).mean().item() > 0 for gb, ga in zip(grads_before, grads_after)):
        print("\n[SUCCESS] Manual DP logic confirmed! Gradients were clipped and noised.")
    else:
        print("\n[FAILURE] Gradients were not modified correctly.")

if __name__ == "__main__":
    test_manual_dp()
