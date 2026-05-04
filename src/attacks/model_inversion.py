import os
import sys
import zipfile
import tempfile
import urllib.request
import gc
import gdown
from typing import Optional

# Path and Cache Configuration
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force PyTorch Hub to download models to the F: drive instead of C: drive
os.environ["TORCH_HOME"] = os.path.join(_REPO_ROOT, "cns_project_cache", "torch_home")
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image

# -----------------------------
# MobileStyleGAN CPU Inversion (Lite)
# -----------------------------
class MobileStyleGANLite(nn.Module):
    """Memory-efficient wrapper for MobileStyleGAN generator."""
    def __init__(self, mapping_net, student, style_mean):
        super().__init__()
        self.mapping_net = mapping_net
        self.student = student
        self.register_buffer("style_mean", style_mean)
        
    def forward(self, var=None, style=None, truncated=False, generator="student"):
        # Support direct style (W+) injection or MappingNet (Z) input
        if style is None:
            style = self.mapping_net(var)
            if truncated:
                # Simple truncation as implemented in Distiller
                style = self.style_mean + 0.5 * (style - self.style_mean)
        
        return self.student(style)["img"]

def load_mobilestylegan(device):
    """Memory-optimized loader for MobileStyleGAN.
    Saves RAM by skipping Teacher, VGG16, and Inception models.
    """
    mob_dir = _ensure_mobilestylegan_repo()
    if not mob_dir:
        return None

    if mob_dir not in sys.path:
        sys.path.insert(0, mob_dir)
        
    try:
        from core.utils import load_cfg, select_weights, load_weights
        from core.model_zoo import model_zoo
        from core.models.mapping_network import MappingNetwork
        from core.models.mobile_synthesis_network import MobileSynthesisNetwork
    except ImportError as e:
        print(f"    [WARNING] Failed to import MobileStyleGAN components: {e}")
        return None

    print("    Loading MobileStyleGAN Lite (RAM Optimized)...")
    cfg_path = os.path.join(mob_dir, "configs", "mobile_stylegan_ffhq.json")
    if not os.path.exists(cfg_path):
        print(f"    [WARNING] Config file not found: {cfg_path}")
        return None
    
    cfg = load_cfg(cfg_path)
    
    # 1. Load the checkpoint once
    print("    Loading weights from disk...")
    ckpt_path = os.path.join(mob_dir, "mobilestylegan_ffhq.ckpt")
    
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
    else:
        print(f"    [INFO] Downloading pretrained weights to {ckpt_path}...")
        gdown.download(id="11Kja0XGE8liLb6R5slNZjF3j3v_6xydt", output=ckpt_path, quiet=False)
        ckpt = torch.load(ckpt_path, map_location="cpu")
    
    state_dict = ckpt["state_dict"]
    
    # 2. Initialize minimal networks
    # MappingNetwork params: style_dim=512, n_layers=8 (standard for FFHQ)
    print("    Initializing Mapping Network...")
    mapping_net = MappingNetwork(style_dim=512, n_layers=8).to(device)
    mapping_weights = select_weights(state_dict, prefix="mapping_net.")
    load_weights(mapping_net, mapping_weights)
    
    # Student SynthesisNetwork
    print("    Initializing Student Synthesis Network...")
    student = MobileSynthesisNetwork(style_dim=512).to(device)
    student_weights = select_weights(state_dict, prefix="student.")
    load_weights(student, student_weights)
    
    style_mean = state_dict["style_mean"].to(device)
    
    # 3. CRITICAL: Clear large checkpoint objects from RAM immediately
    del ckpt
    del state_dict
    gc.collect()
    
    # 4. Wrap it
    generator = MobileStyleGANLite(mapping_net, student, style_mean).to(device)
    generator.eval()
    for param in generator.parameters():
        param.requires_grad = False
        
    return generator

def total_variation_loss(img):
    """Calculates Total Variation loss to encourage smoothness."""
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w)

def symmetry_loss(img):
    """Encourages the face to be front-facing by rewarding horizontal symmetry."""
    flipped_img = torch.flip(img, [3])
    return F.mse_loss(img, flipped_img)

def run_mobilestylegan_inversion_attack(
    model,
    target_embedding: torch.Tensor,
    generator,
    iterations: int = 1000,
    lr: float = 0.005,
    save_path: str = None
):
    """MobileStyleGAN latent inversion optimized for CPU."""
    device = next(model.parameters()).device
    
    # High-Fidelity W+ Optimization
    # 1. Start from the 'Average Face'
    # We use W+ [1, 23, 512] for fine-grained detail (eyes, pose, features)
    w_avg = generator.style_mean.detach().clone() # [1, 512]
    w_plus = w_avg.unsqueeze(1).repeat(1, 23, 1).detach().clone().requires_grad_(True)
    
    optimizer = torch.optim.Adam([w_plus], lr=lr)
    
    print(f"    Running High-Fidelity MobileStyleGAN attack on {device}...")
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Generate image from current latent
        img_s = generator(style=w_plus)
        
        # Noise Augmentation: Prevents optimization from getting stuck in 
        # local minima and encourages sharper, more robust facial features.
        if i < iterations * 0.8:
            # Add 1% noise during the search phase
            noise = torch.randn_like(img_s) * 0.01
            img_proc = img_s + noise
        else:
            # Clean image for the final refinement phase
            img_proc = img_s

        # Resize to FaceNet input size (160x160)
        synth_img = F.interpolate(img_proc, size=(160, 160), mode='bilinear', align_corners=False)
        
        generated_emb = model(synth_img)
        
        # Multi-part Loss function
        # 1. Identity loss (Primary goal)
        identity_loss = 1.0 - F.cosine_similarity(generated_emb, target_embedding.detach(), dim=1).mean()
        
        # 2. W+ Regularization (Stay close to 'Human' manifold)
        w_reg = torch.mean((w_plus - w_avg.unsqueeze(1)).pow(2))
        
        # 3. TV Loss (Prevent noise/blurriness)
        tv = total_variation_loss(img_s)
        
        # 4. Symmetry Loss (Force front-facing pose)
        sym = symmetry_loss(img_s)
        
        # Combine losses with weights tuned for CPU/8GB stability
        # Maximizing sharpness: High identity weight, low regularization, low TV
        loss = 20.0 * identity_loss + 0.5 * w_reg + 0.00005 * tv + 0.05 * sym
        
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"      iter {i:4d}: identity={identity_loss.item():.4f}, sym={sym.item():.4f}, w_reg={w_reg.item():.4f}")
            gc.collect()

    final_image_batch = generator(style=w_plus.detach()).detach()
    # Resize for saving so it matches the expected 160x160 output
    final_image = F.interpolate(final_image_batch, size=(160, 160), mode='bilinear', align_corners=False)
    
    if save_path:
        from torchvision.utils import save_image
        img_save = (final_image.squeeze() + 1.0) / 2.0
        save_image(img_save, save_path)
        
    return final_image, loss.item()

FACENET_INPUT = 160


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    resized = image.resize((FACENET_INPUT, FACENET_INPUT), Image.BILINEAR)
    array = torch.from_numpy(__import__("numpy").array(resized, dtype="float32")).permute(2, 0, 1)
    return (array / 127.5 - 1.0).unsqueeze(0)


def _ensure_image_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-2:] != (FACENET_INPUT, FACENET_INPUT):
        tensor = F.interpolate(tensor, size=(FACENET_INPUT, FACENET_INPUT), mode="bilinear", align_corners=False)
    return tensor.float().clamp(-1.0, 1.0)

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
            return _ensure_image_tensor(tensor)

        image = Image.open(file_path).convert("RGB")
        return _pil_to_tensor(image)

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

    for file_path in selected_paths:
        if file_path.lower().endswith(".pt"):
            tensor = torch.load(file_path, map_location="cpu").float()
            image_tensors.append(_ensure_image_tensor(tensor))
        else:
            image = Image.open(file_path).convert("RGB")
            image_tensors.append(_pil_to_tensor(image))

    prior = torch.mean(torch.cat(image_tensors, dim=0), dim=0, keepdim=True)
    return prior.clamp(-1.0, 1.0)


def get_general_face_prior(clients_root: str, exclude_client: str = None, max_images: int = 16) -> torch.Tensor:
    """
    Build a fair 'General Ghost' prior by averaging faces from ALL OTHER clients.
    This simulates an attacker who knows what humans look like generally, but has 
    zero knowledge of the specific target.
    """
    all_face_paths = []
    
    # Walk through all clients
    for client_name in os.listdir(clients_root):
        if exclude_client and client_name == exclude_client:
            continue
            
        client_path = os.path.join(clients_root, client_name)
        if not os.path.isdir(client_path):
            continue
            
        for root, _, files in os.walk(client_path):
            for file_name in files:
                if file_name.lower().endswith((".pt", ".jpg", ".jpeg", ".png")):
                    all_face_paths.append(os.path.join(root, file_name))
    
    if not all_face_paths:
        # Fallback to random noise if no other data exists
        return torch.randn(1, 3, 160, 160) * 0.1
        
    # Pick a diverse set of images
    import random
    random.shuffle(all_face_paths)
    selected_paths = all_face_paths[:max_images]
    
    image_tensors = []
    for path in selected_paths:
        try:
            if path.lower().endswith(".pt"):
                tensor = torch.load(path, map_location="cpu").float()
                image_tensors.append(_ensure_image_tensor(tensor))
            else:
                image = Image.open(path).convert("RGB")
                image_tensors.append(_pil_to_tensor(image))
        except Exception:
            continue
            
    if not image_tensors:
        return torch.randn(1, 3, 160, 160) * 0.1
        
    return torch.mean(torch.cat(image_tensors, dim=0), dim=0, keepdim=True).clamp(-1.0, 1.0)


def load_stylegan_generator(network_pkl: str, stylegan_repo_dir: Optional[str] = None, device: str = "cpu"):
    """Load a pretrained StyleGAN2/StyleGAN3 generator from an official NVLabs pickle."""
    repo_dir = stylegan_repo_dir
    if not repo_dir or not os.path.exists(repo_dir):
        repo_dir = _ensure_stylegan_repo()

    if repo_dir and repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    # Force StyleGAN to use the F: drive cache for weights and compiled code
    os.environ["DNNLIB_CACHE_DIR"] = os.path.join(_REPO_ROOT, "cns_project_cache")
    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(_REPO_ROOT, "cns_project_cache", "torch_extensions")

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


def _ensure_mobilestylegan_repo() -> Optional[str]:
    """Download and cache MobileStyleGAN repository if not available."""
    cache_root = os.path.join(_REPO_ROOT, "cns_project_cache")
    repo_root = os.path.join(cache_root, "MobileStyleGAN")
    
    # Check if a specific file exists to verify the repo is complete
    if os.path.exists(os.path.join(repo_root, "core", "utils.py")):
        return repo_root

    os.makedirs(cache_root, exist_ok=True)
    url = "https://github.com/bes-dev/MobileStyleGAN.pytorch/archive/refs/heads/develop.zip"
    archive_path = os.path.join(cache_root, "MobileStyleGAN.zip")

    print(f"    [INFO] Downloading MobileStyleGAN repository from {url}...")
    try:
        import shutil
        urllib.request.urlretrieve(url, archive_path)
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(cache_root)
        
        # The zip extracts to MobileStyleGAN.pytorch-develop, rename it to MobileStyleGAN
        extracted_folder = os.path.join(cache_root, "MobileStyleGAN.pytorch-develop")
        if os.path.exists(extracted_folder):
            if os.path.exists(repo_root):
                shutil.rmtree(repo_root)
            os.rename(extracted_folder, repo_root)
        
        if os.path.exists(archive_path):
            os.remove(archive_path)
            
        print("    [INFO] MobileStyleGAN repository setup complete.")
        return repo_root
    except Exception as e:
        print(f"    [ERROR] Failed to download MobileStyleGAN repo: {e}")
        return None


def _ensure_stylegan_repo() -> str:
    """Download and cache StyleGAN2-ADA locally if it is not already available."""
    cache_root = os.path.join(_REPO_ROOT, "cns_project_cache")
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

def _embedding_from_image(model, image_path: str, device: str):
    """
    Get embedding and tensor from a raw image file.
    Uses MTCNN to detect and align the face (160x160) before embedding.
    Falls back to simple resize if no face is detected.
    """
    lower = image_path.lower()
    if lower.endswith(".pt"):
        tensor = torch.load(image_path, map_location="cpu").float()
        tensor = _ensure_image_tensor(tensor)
        name = os.path.splitext(os.path.basename(image_path))[0]
    else:
        raw_image = Image.open(image_path).convert("RGB")
        name = os.path.splitext(os.path.basename(image_path))[0]

        # --- MTCNN face detection + alignment ---
        try:
            from facenet_pytorch import MTCNN
            mtcnn = MTCNN(image_size=FACENET_INPUT, margin=20, keep_all=False,
                          device=device, post_process=False)
            face_tensor = mtcnn(raw_image)  # returns [3, 160, 160] float or None
            if face_tensor is not None:
                # mtcnn returns 0-255 floats; normalise to [-1, 1]
                face_tensor = (face_tensor / 127.5 - 1.0).unsqueeze(0)
                tensor = face_tensor.clamp(-1.0, 1.0)
                print(f"    [MTCNN] Face detected and cropped from {os.path.basename(image_path)}")
            else:
                print(f"    [MTCNN] No face detected in {os.path.basename(image_path)}, using full image resize.")
                tensor = _pil_to_tensor(raw_image)
        except ImportError:
            print("    [WARNING] facenet_pytorch not installed; using simple resize.")
            tensor = _pil_to_tensor(raw_image)

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        embedding = model(tensor.to(device))

    return embedding, name, tensor


def attack_both_models(
    model_no_dp_path: str,
    model_with_dp_path: str,
    client_dir: str,
    output_dir: str,
    iterations: int = 1000,
    attack_lr: float = 0.02,
    plot_file_tag: str = "attack",
    target_image_path: Optional[str] = None,
) -> dict:
    """
    Run inversion attack on BOTH models using the same target.
    Always runs BOTH MobileStyleGAN (latent) and Pixel-Space attacks.
    Pixel-space always starts from the GAN ghost face (zero-knowledge).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading MobileStyleGAN on {device}...")

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    generator = load_mobilestylegan(device)

    # -- Ghost face prior: decode GAN style_mean → pixel image (zero-knowledge) --
    if generator is not None:
        with torch.no_grad():
            w_avg      = generator.style_mean.detach()              # [1, 512]
            w_avg_plus = w_avg.unsqueeze(1).repeat(1, 23, 1)        # [1, 23, 512]
            ghost_face = generator(style=w_avg_plus).detach().cpu() # [1, 3, H, W]
            ghost_face = ghost_face.clamp(-1.0, 1.0)
    else:
        ghost_face = torch.zeros(1, 3, 160, 160)  # neutral gray

    # -- 1. Attack No-DP Model --
    print("  Running on Version A (no DP)...")
    model_no_dp = load_model_from_checkpoint(model_no_dp_path).to(device)

    if target_image_path:
        target_emb_no_dp, person_name, original_tensor = _embedding_from_image(
            model_no_dp, target_image_path, device
        )
    else:
        target_emb_no_dp, person_name, original_tensor, _ = get_target_embedding(
            model_no_dp, client_dir
        )

    # 1a. MobileStyleGAN attack
    if generator:
        path_msg_no_dp = os.path.join(output_dir, f"msg_{plot_file_tag}_no_dp.png")
        fake_msg_no_dp, loss_msg_no_dp = run_mobilestylegan_inversion_attack(
            model_no_dp, target_emb_no_dp,
            generator=generator, iterations=iterations, lr=attack_lr,
            save_path=path_msg_no_dp
        )
        print(f"  Saved: {path_msg_no_dp}")
    else:
        fake_msg_no_dp, loss_msg_no_dp = None, None

    # 1b. Pixel-space attack (always, starts from ghost face)
    path_pix_no_dp = os.path.join(output_dir, f"pixel_{plot_file_tag}_no_dp.png")
    fake_pix_no_dp, loss_pix_no_dp = run_inversion_attack(
        model_no_dp, target_emb_no_dp,
        iterations=iterations, lr=attack_lr,
        start_tensor=ghost_face.to(device),
        save_path=path_pix_no_dp
    )
    print(f"  Saved: {path_pix_no_dp}")

    del model_no_dp
    gc.collect()

    # -- 2. Attack With-DP Model --
    print("  Running on Version B (with DP)...")
    model_with_dp = load_model_from_checkpoint(model_with_dp_path).to(device)

    if target_image_path:
        target_emb_dp, _, _ = _embedding_from_image(
            model_with_dp, target_image_path, device
        )
    else:
        target_emb_dp, _, _, _ = get_target_embedding(model_with_dp, client_dir)

    # 2a. MobileStyleGAN attack
    if generator:
        path_msg_dp = os.path.join(output_dir, f"msg_{plot_file_tag}_with_dp.png")
        fake_msg_dp, loss_msg_dp = run_mobilestylegan_inversion_attack(
            model_with_dp, target_emb_dp,
            generator=generator, iterations=iterations, lr=attack_lr,
            save_path=path_msg_dp
        )
        print(f"  Saved: {path_msg_dp}")
    else:
        fake_msg_dp, loss_msg_dp = None, None

    # 2b. Pixel-space attack (always, starts from ghost face)
    path_pix_dp = os.path.join(output_dir, f"pixel_{plot_file_tag}_with_dp.png")
    fake_pix_dp, loss_pix_dp = run_inversion_attack(
        model_with_dp, target_emb_dp,
        iterations=iterations, lr=attack_lr,
        start_tensor=ghost_face.to(device),
        save_path=path_pix_dp
    )
    print(f"  Saved: {path_pix_dp}")

    del model_with_dp
    gc.collect()

    # -- 3. Build 5-panel comparison figure --
    n_cols = 5 if generator else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    def _show(ax, tensor, title, color="black"):
        img = (tensor.squeeze().cpu() + 1) / 2
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(title, color=color, fontsize=9)
        ax.axis("off")

    col = 0
    _show(axes[col], original_tensor, f"Original\n({person_name})")
    col += 1

    if generator:
        _show(axes[col], fake_msg_no_dp,
              "MobileStyleGAN\nNo DP  ⚠️", color="red")
        col += 1
        _show(axes[col], fake_pix_no_dp,
              "Pixel-Space\nNo DP  ⚠️", color="red")
        col += 1
        _show(axes[col], fake_msg_dp,
              "MobileStyleGAN\nWith DP  🔒", color="green")
        col += 1
        _show(axes[col], fake_pix_dp,
              "Pixel-Space\nWith DP  🔒", color="green")
    else:
        _show(axes[col], fake_pix_no_dp,
              "Pixel-Space\nNo DP  ⚠️", color="red")
        col += 1
        _show(axes[col], fake_pix_dp,
              "Pixel-Space\nWith DP  🔒", color="green")

    plt.suptitle("Model Inversion Attack Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, f"comparison_{plot_file_tag}.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {comparison_path}")

    return {
        "client_tag":       plot_file_tag,
        "msg_no_dp_loss":   loss_msg_no_dp,
        "pix_no_dp_loss":   loss_pix_no_dp,
        "msg_dp_loss":      loss_msg_dp,
        "pix_dp_loss":      loss_pix_dp,
    }
