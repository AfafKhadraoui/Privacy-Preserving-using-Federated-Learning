import os
import sys
import zipfile
import tempfile
import urllib.request
import gc
from typing import Optional
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

try:
    import lpips
except Exception:
    lpips = None

# Ensure config is available
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import CROPPED_DIR, STYLEGAN_REPO_URL
from src.model.face_model import get_model

def load_model_from_checkpoint(checkpoint_path: str):
    """
    Load InceptionResnetV1 from a .pth checkpoint file.
    Handles both DP-wrapped and non-DP models by stripping DP-specific state keys.
    """
    model = get_model(mode="eval")
    # map_location handles environments without a GPU
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # Strip DP wrapper prefixes if present (opacus wraps module with "._module.")
    cleaned_state = {}
    for k, v in state_dict.items():
        # Remove ._module. prefix if it exists (from Opacus PrivacyEngine)
        if k.startswith("_module."):
            cleaned_state[k[8:]] = v  # Remove "_module." prefix
        else:
            cleaned_state[k] = v
    
    model.load_state_dict(cleaned_state, strict=False)
    model.eval()
    
    # Explicitly remove all hooks (may be left from DP wrapping)
    model._backward_hooks = None
    for module in model.modules():
        module._forward_hooks = {}
        module._backward_hooks = {}
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def get_target_embedding(model, client_dir: str) -> tuple[torch.Tensor, str]:
    """
    Load a face tensor or image from client_dir, run through model, return embedding.
    """
    person_name = "Unknown"

    def _load_face_file(file_path: str) -> torch.Tensor:
        lower = file_path.lower()
        if lower.endswith(".pt"):
            tensor = torch.load(file_path, map_location="cpu")
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            return tensor.float()

        image = Image.open(file_path).convert("RGB")
        image_transform = T.Compose([
            T.Resize((160, 160)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return image_transform(image).unsqueeze(0)

    # Search only one level: find first person directory, then first face file inside it.
    person_dirs = [d for d in os.listdir(client_dir) if os.path.isdir(os.path.join(client_dir, d))]
    if person_dirs:
        person_name = person_dirs[0]
        person_dir = os.path.join(client_dir, person_name)
        face_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.pt', '.jpg', '.jpeg', '.png'))]
        if not face_files:
            raise ValueError(f"No face files found in {person_dir}")
        target_tensor_path = os.path.join(person_dir, face_files[0])
        tensor = _load_face_file(target_tensor_path)
    else:
        face_files = [f for f in os.listdir(client_dir) if f.lower().endswith(('.pt', '.jpg', '.jpeg', '.png'))]
        if not face_files:
            raise ValueError(f"No person directories or face files found in {client_dir}")
        target_tensor_path = os.path.join(client_dir, face_files[0])
        tensor = _load_face_file(target_tensor_path)

    # Tensor shape is expected to be [1, 3, 160, 160] at this point.
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        
    model.eval()
    with torch.no_grad():
        embedding = model(tensor)
        
    return embedding, person_name, tensor, target_tensor_path


def load_face_prior(prior_dir: str, max_images: int = 32) -> torch.Tensor:
    """Build a simple face prior by averaging available face tensors or images."""
    candidate_paths = []
    for root, _, files in os.walk(prior_dir):
        for file_name in files:
            lower_name = file_name.lower()
            if lower_name.endswith((".pt", ".jpg", ".jpeg", ".png")):
                candidate_paths.append(os.path.join(root, file_name))

    if not candidate_paths:
        raise ValueError(f"No face files found in {prior_dir}")

    candidate_paths.sort()
    selected_paths = candidate_paths[:max_images]

    image_tensors = []
    image_transform = T.Compose([
        T.Resize((160, 160)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    for file_path in selected_paths:
        if file_path.lower().endswith(".pt"):
            tensor = torch.load(file_path, map_location="cpu").float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[-2:] != (160, 160):
                tensor = F.interpolate(tensor, size=(160, 160), mode="bilinear", align_corners=False)
            image_tensors.append(tensor.clamp(-1.0, 1.0))
        else:
            image = Image.open(file_path).convert("RGB")
            tensor = image_transform(image).unsqueeze(0)
            image_tensors.append(tensor)

    prior = torch.mean(torch.cat(image_tensors, dim=0), dim=0, keepdim=True)
    return prior.clamp(-1.0, 1.0)


def load_stylegan_generator(network_pkl: str, stylegan_repo_dir: Optional[str] = None, device: str = "cpu"):
    """Load a pretrained StyleGAN2/StyleGAN3 generator from an official NVLabs pickle."""
    repo_dir = stylegan_repo_dir
    if not repo_dir or not os.path.exists(repo_dir):
        repo_dir = _ensure_stylegan_repo()

    if repo_dir and repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    try:
        import dnnlib
        import legacy
    except Exception as exc:
        raise ImportError(
            "StyleGAN loading requires the NVLabs StyleGAN repo code on sys.path."
        ) from exc

    with dnnlib.util.open_url(network_pkl) as fp:
        network_data = legacy.load_network_pkl(fp)

    generator = network_data["G_ema"].to(device)
    generator.eval()
    for param in generator.parameters():
        param.requires_grad = False
    return generator


def _ensure_stylegan_repo() -> str:
    """Download and cache StyleGAN2-ADA locally if it is not already available."""
    cache_root = os.path.join(tempfile.gettempdir(), "stylegan2-ada-pytorch-cache")
    repo_root = os.path.join(cache_root, "stylegan2-ada-pytorch-main")
    if os.path.exists(os.path.join(repo_root, "dnnlib")):
        return repo_root

    os.makedirs(cache_root, exist_ok=True)
    archive_path = os.path.join(cache_root, "stylegan2-ada-pytorch-main.zip")

    if not os.path.exists(archive_path):
        with urllib.request.urlopen(STYLEGAN_REPO_URL) as response, open(archive_path, "wb") as handle:
            handle.write(response.read())

    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(cache_root)

    return repo_root


def _stylegan_synthesize(generator, latent: torch.Tensor) -> torch.Tensor:
    """Synthesize an image from a StyleGAN generator, handling z-space and w-space generators."""
    if hasattr(generator, "mapping") and hasattr(generator, "synthesis"):
        class_labels = None
        if hasattr(generator, "c_dim") and getattr(generator, "c_dim", 0) > 0:
            class_labels = torch.zeros([latent.shape[0], generator.c_dim], device=latent.device)

        if latent.dim() == 2:
            ws = generator.mapping(latent, class_labels)
        else:
            ws = latent
        if ws.dim() == 2:
            ws = ws.unsqueeze(1)
        image = generator.synthesis(ws, noise_mode="const")
        if isinstance(image, (tuple, list)):
            image = image[0]
        return image.clamp(-1.0, 1.0)

    raise TypeError("Unsupported StyleGAN generator interface")


def run_stylegan_inversion_attack(
    model,
    target_embedding: torch.Tensor,
    generator,
    iterations: int = 1500,
    lr: float = 0.01,
    identity_weight: float = 1.0,
    perceptual_weight: float = 0.1,
    latent_reg_weight: float = 0.001,
    reference_image: torch.Tensor = None,
    save_path: str = None,
    seed: int = None,
):
    """StyleGAN-based inversion attack in latent space."""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if seed is not None:
        torch.manual_seed(seed)

    device = next(generator.parameters()).device
    latent_dim = getattr(generator, "z_dim", None) or getattr(generator, "w_dim", 512)
    latent = torch.randn(1, latent_dim, device=device) * 0.5
    latent.requires_grad_(True)

    optimizer = torch.optim.Adam([latent], lr=lr)

    perceptual_fn = None
    if perceptual_weight > 0 and lpips is not None and reference_image is not None:
        perceptual_fn = lpips.LPIPS(net="vgg").to(device).eval()

    loss_history = []
    with torch.no_grad():
        try:
            start_image = _stylegan_synthesize(generator, latent)
            start_emb = model(start_image)
            initial_dist = float(torch.norm(start_emb - target_embedding).item())
        except Exception:
            initial_dist = None

    for i in range(iterations):
        optimizer.zero_grad()

        synth_img = _stylegan_synthesize(generator, latent)
        generated_emb = model(synth_img)

        identity_loss = 1.0 - F.cosine_similarity(generated_emb, target_embedding.detach(), dim=1).mean()

        perceptual_loss = torch.tensor(0.0, device=device)
        if perceptual_fn is not None:
            perceptual_loss = perceptual_fn(synth_img, reference_image.to(device)).mean()

        latent_reg = torch.mean(latent.pow(2))

        loss = (
            identity_weight * identity_loss
            + perceptual_weight * perceptual_loss
            + latent_reg_weight * latent_reg
        )

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            latent.clamp_(-3.0, 3.0)

        loss_history.append(float(loss.item()))

        if i % 200 == 0:
            print(
                f"    iter {i:4d}: identity={identity_loss.item():.6f}, "
                f"perc={perceptual_loss.item():.6f}, latent={latent_reg.item():.6f}, total={loss.item():.6f}"
            )

    final_image = _stylegan_synthesize(generator, latent).detach()

    if save_path:
        try:
            from torchvision.utils import save_image
            img_save = (final_image.squeeze() + 1.0) / 2.0
            save_image(img_save, save_path)
        except Exception:
            pass

    return final_image, loss.item(), {"initial_distance": initial_dist, "loss_history": loss_history}

def run_inversion_attack(
    model,
    target_embedding: torch.Tensor,
    iterations: int = 2000,
    lr: float = 0.01,
    start_tensor: torch.Tensor = None,
    save_path: str = None,
    save_loss_path: str = None,
    seed: int = None,
) -> torch.Tensor:
    """
    Run model inversion attack.

    Strategy: feature-space attack with face prior. Start optimization
    from the provided start_tensor if available; otherwise use small
    gaussian noise. Records loss history and initial embedding distance
    when requested.

    Returns reconstructed image tensor [1, 3, 160, 160].
    """
    model.eval()
    for param in model.parameters(): 
        param.requires_grad = False

    # Deterministic noise if seed provided
    if seed is not None:
        torch.manual_seed(seed)

    # Use provided start tensor or small Gaussian noise
    if start_tensor is not None:
        fake_img = start_tensor.clone().float()
    else:
        fake_img = torch.randn(1, 3, 160, 160) * 0.1

    fake_img.requires_grad_(True)
    optimizer = torch.optim.Adam([fake_img], lr=lr)
    criterion = nn.MSELoss()

    loss_history = []
    # record initial distance between start and target embedding
    with torch.no_grad():
        try:
            start_emb = model(fake_img)
            initial_dist = float(torch.norm(start_emb - target_embedding).item())
        except Exception:
            initial_dist = None

    for i in range(iterations):
        optimizer.zero_grad()
        fake_emb = model(fake_img)

        mse = criterion(fake_emb, target_embedding.detach())

        # VERY small TV weight — just enough to reduce checkerboard artifacts
        tv = (
            torch.mean(torch.abs(fake_img[:, :, :, :-1] - fake_img[:, :, :, 1:])) +
            torch.mean(torch.abs(fake_img[:, :, :-1, :] - fake_img[:, :, 1:, :]))
        )

        loss = mse + 1e-6 * tv

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            fake_img.clamp_(-1.0, 1.0)

        loss_history.append(float(loss.item()))

        if i % 200 == 0:
            print(f"    iter {i:4d}: mse={mse.item():.6f}, tv={tv.item():.6f}, total={loss.item():.6f}")
        
        # Clear intermediate tensors to save memory
        if i % 50 == 0:
            del fake_emb, mse, tv, loss
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if save_path:
        try:
            from torchvision.utils import save_image
            img_save = (fake_img.detach().squeeze() + 1.0) / 2.0
            save_image(img_save, save_path)
        except Exception:
            pass

    # Optionally save loss history and initial distance
    if save_loss_path:
        try:
            import json
            os.makedirs(os.path.dirname(save_loss_path), exist_ok=True)
            with open(save_loss_path, 'w') as fh:
                json.dump({'initial_distance': initial_dist, 'loss_history': loss_history}, fh)
        except Exception:
            pass

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
    target_emb_no_dp, person_name, original_tensor, _ = get_target_embedding(model_no_dp, client_dir)
    
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
    target_emb_with_dp, _, _, _ = get_target_embedding(model_with_dp, client_dir)
    
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
