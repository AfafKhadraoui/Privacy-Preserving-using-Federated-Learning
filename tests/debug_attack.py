import torch
import sys
import os
import gc

# Add project root to path
sys.path.append(os.getcwd())

from src.attacks.model_inversion import load_mobilestylegan, get_model

def debug_attack_step():
    print("--- Debugging Attack Step ---", flush=True)
    device = torch.device("cpu")
    
    print("1. Loading FaceNet...", flush=True)
    model = get_model(mode="eval").to(device)
    
    print("2. Loading MobileStyleGAN...", flush=True)
    generator = load_mobilestylegan(device)
    
    print("3. Initializing W+ latent...", flush=True)
    w_avg = generator.style_mean.detach().clone()
    w_plus = w_avg.unsqueeze(1).repeat(1, 23, 1).detach().clone().requires_grad_(True)
    
    print(f"   Latent shape: {w_plus.shape}", flush=True)

    print("4. Testing Generator Forward Pass...", flush=True)
    try:
        img = generator(style=w_plus)
        print(f"   Success! Image shape: {img.shape}", flush=True)
    except Exception as e:
        print(f"   FAILED at Generator: {e}", flush=True)
        return

    print("5. Testing Model Forward Pass...", flush=True)
    try:
        img_resized = torch.nn.functional.interpolate(img, size=(160, 160))
        emb = model(img_resized)
        print(f"   Success! Embedding shape: {emb.shape}", flush=True)
    except Exception as e:
        print(f"   FAILED at Model: {e}", flush=True)
        return

    print("6. Testing Backward Pass (Memory Intensive)...", flush=True)
    try:
        loss = emb.sum()
        loss.backward()
        print("   Success! Backward pass complete.", flush=True)
    except Exception as e:
        print(f"   FAILED at Backward: {e}", flush=True)
        return

    print("\n--- ALL STEPS PASSED ---", flush=True)

if __name__ == "__main__":
    debug_attack_step()
