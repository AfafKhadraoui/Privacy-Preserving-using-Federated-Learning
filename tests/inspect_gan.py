import torch
import sys
import os

sys.path.append(os.getcwd())
from src.attacks.model_inversion import load_mobilestylegan

def inspect_gan():
    gen = load_mobilestylegan("cpu")
    print("\n--- MobileStyleGAN Structure ---")
    
    # Check student layers
    student = gen.student
    print(f"Student Class: {type(student)}")
    
    # Try to find style depth
    # We can look at the input shape of the first synthesis block or similar
    if hasattr(student, 'num_layers'):
        print(f"Layers (num_layers): {student.num_layers}")
    
    # Test different W+ depths
    for d in [8, 12, 14, 16, 18, 23]:
        print(f"Testing W+ depth {d}...", end=" ", flush=True)
        try:
            w = torch.randn(1, d, 512)
            out = student(w)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

if __name__ == "__main__":
    inspect_gan()
