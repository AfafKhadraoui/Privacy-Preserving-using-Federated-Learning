import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Ensure config is available
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.face_model import get_model

def load_model_from_checkpoint(checkpoint_path: str):
    """
    Load InceptionResnetV1 from a .pth checkpoint file.
    """
    model = get_model(mode="eval")
    # map_location handles environments without a GPU
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def get_target_embedding(model, client_dir: str) -> tuple[torch.Tensor, str]:
    """
    Load a face tensor from client_dir, run through model, return embedding.
    """
    # Find the first .pt file in the directory
    target_tensor_path = None
    person_name = "Unknown"
    for root, dirs, files in os.walk(client_dir):
        for file in files:
            if file.endswith(".pt"):
                target_tensor_path = os.path.join(root, file)
                person_name = os.path.basename(root)
                break
        if target_tensor_path:
            break
            
    if not target_tensor_path:
        raise ValueError(f"No .pt tensor files found in {client_dir}")
        
    tensor = torch.load(target_tensor_path, map_location="cpu")
    # Tensor shape is expected to be [3, 160, 160]. We need to unsqueeze to add batch dim
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
        
    model.eval()
    with torch.no_grad():
        embedding = model(tensor)
        
    return embedding, person_name, tensor

def run_inversion_attack(
    model,
    target_embedding: torch.Tensor,
    iterations: int = 1000,
    lr: float = 0.01,
    save_path: str = None
) -> torch.Tensor:
    """
    Run model inversion attack.
    Returns reconstructed image tensor [1, 3, 160, 160].
    """
    model.eval()
    for param in model.parameters(): 
        param.requires_grad = False
        
    # Initialize fake image with random noise
    fake_img = torch.randn(1, 3, 160, 160, requires_grad=True)
    optimizer = torch.optim.Adam([fake_img], lr=lr)
    criterion = nn.MSELoss()
    
    for i in range(iterations):
        optimizer.zero_grad()
        fake_emb = model(fake_img)
        loss = criterion(fake_emb, target_embedding)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad(): 
            fake_img.clamp_(-1.0, 1.0)
            
        if i % 100 == 0: 
            print(f"    iter {i:4d}: loss={loss.item():.4f}")
            
    if save_path:
        # Save the result image
        img_to_save = (fake_img.detach().squeeze() + 1) / 2
        import torchvision
        torchvision.utils.save_image(img_to_save, save_path)
        
    return fake_img.detach(), loss.item()

def attack_both_models(
    model_no_dp_path: str,
    model_with_dp_path: str,
    client_dir: str,
    output_dir: str,
    iterations: int = 1000
) -> dict:
    """
    Run inversion attack on BOTH models using the same target.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Attack No DP Model
    print("  Running on Version A (no DP)...")
    model_no_dp = load_model_from_checkpoint(model_no_dp_path)
    target_emb_no_dp, person_name, original_tensor = get_target_embedding(model_no_dp, client_dir)
    
    path_no_dp = os.path.join(output_dir, "attack_no_dp.png")
    fake_img_no_dp, final_loss_no_dp = run_inversion_attack(
        model_no_dp, 
        target_emb_no_dp, 
        iterations=iterations, 
        save_path=path_no_dp
    )
    print(f"  Saved: {path_no_dp}")
    
    # 2. Attack With DP Model
    print("  Running on Version B (with DP)...")
    model_with_dp = load_model_from_checkpoint(model_with_dp_path)
    target_emb_with_dp, _, _ = get_target_embedding(model_with_dp, client_dir)
    
    path_with_dp = os.path.join(output_dir, "attack_with_dp.png")
    fake_img_with_dp, final_loss_with_dp = run_inversion_attack(
        model_with_dp, 
        target_emb_with_dp, 
        iterations=iterations, 
        save_path=path_with_dp
    )
    print(f"  Saved: {path_with_dp}")
    
    # 3. Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # original_tensor is [1, 3, 160, 160] with values in [-1, 1]
    original_img = (original_tensor.squeeze() + 1) / 2
    axes[0].imshow(original_img.permute(1,2,0).numpy())
    axes[0].set_title(f"Original Face ({person_name})")
    axes[0].axis("off")
    
    attack_no_dp = (fake_img_no_dp.squeeze() + 1) / 2
    axes[1].imshow(attack_no_dp.permute(1,2,0).numpy())
    axes[1].set_title("Attack Result\n(No DP — Face Visible)", color="red")
    axes[1].axis("off")
    
    attack_with_dp = (fake_img_with_dp.squeeze() + 1) / 2
    axes[2].imshow(attack_with_dp.permute(1,2,0).numpy())
    axes[2].set_title("Attack Result\n(With DP — Protected)", color="green")
    axes[2].axis("off")
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "attack_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved: {comparison_path}")
    
    return {
        "no_dp_final_loss": final_loss_no_dp,
        "with_dp_final_loss": final_loss_with_dp
    }
