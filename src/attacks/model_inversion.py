import os
import sys
import zipfile
import tempfile
import hashlib
import urllib.request
from typing import Optional
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

# Ensure config is available
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    CROPPED_DIR,
    STYLEGAN_REPO_URL,
    STYLEGAN_NETWORK_PKL,
    STYLEGAN_REPO_DIR,
    STYLEGAN_IDENTITY_W,
    STYLEGAN_LATENT_REG_W,
)
from src.model.face_model import get_model

FACENET_INPUT = 160
STYLEGAN_DOWNLOAD_TIMEOUT_S = 60


def _downsample_for_facenet(x: torch.Tensor, size: int = FACENET_INPUT) -> torch.Tensor:
    """StyleGAN outputs e.g. 1024²; FaceNet VGGFace2 expects 160²."""
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def _embedding_inversion_loss(pred_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
    """Cosine-based loss in embedding space (stable for L2-normalized face embeddings)."""
    pred_n = F.normalize(pred_emb, dim=1)
    tgt_n = F.normalize(target_emb.detach(), dim=1)
    return (1.0 - (pred_n * tgt_n).sum(dim=1)).mean()

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

def _load_first_client_crop_tensor(client_dir: str) -> tuple[torch.Tensor, str, str]:
    """
    Load the first .pt face crop under client_dir (labs / FL layout).
    Returns (tensor [1,3,160,160] in [-1,1], identity folder name, path).
    """
    if not os.path.isdir(client_dir):
        raise ValueError(f"Not a directory: {client_dir}")
    person_dirs = [d for d in os.listdir(client_dir) if os.path.isdir(os.path.join(client_dir, d))]
    if not person_dirs:
        raise ValueError(f"No person directories found in {client_dir}")
    person_name = person_dirs[0]
    person_path = os.path.join(client_dir, person_name)
    pt_files = sorted(f for f in os.listdir(person_path) if f.endswith(".pt"))
    if not pt_files:
        raise ValueError(f"No .pt tensor files found in {person_path}")
    target_tensor_path = os.path.join(person_path, pt_files[0])
    tensor = torch.load(target_tensor_path, map_location="cpu").float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor.clamp(-1.0, 1.0), person_name, target_tensor_path


def whitebox_target_embedding_from_crop(model, client_dir: str) -> tuple[torch.Tensor, str]:
    """
    Simulates a white-box attacker who only knows E(x) for the victim (no pixels in the optimiser).
    The crop is read once to compute this vector inside this function only; callers must not reuse pixels.
    """
    tensor, person_name, _ = _load_first_client_crop_tensor(client_dir)
    model.eval()
    with torch.no_grad():
        embedding = model(tensor.clone())
    return embedding.detach(), person_name


def evaluator_ground_truth_crop(client_dir: str) -> tuple[torch.Tensor, str]:
    """Load victim crop ONLY for evaluator-side figures — never pass this into inversion losses."""
    tensor, person_name, _ = _load_first_client_crop_tensor(client_dir)
    return tensor.detach(), person_name


def get_target_embedding(model, client_dir: str) -> tuple:
    """Legacy helper: embedding + evaluator crop + path (attack code should prefer whitebox_* instead)."""
    tensor, person_name, path = _load_first_client_crop_tensor(client_dir)
    model.eval()
    with torch.no_grad():
        embedding = model(tensor)
    return embedding, person_name, tensor, path


def load_face_prior(
    prior_dir: str,
    max_images: int = 32,
    exclude_prefixes: Optional[list] = None,
) -> torch.Tensor:
    """
    Build a simple face prior by averaging available tensors/images.
    ``exclude_prefixes``: absolute dirs to skip (exclude victim client so prior is not leaked GT).
    """
    candidate_paths = []
    prefixes = []
    if exclude_prefixes:
        prefixes = [os.path.abspath(p) for p in exclude_prefixes if p]

    def _skip(path: str) -> bool:
        ap = os.path.abspath(path)
        for pref in prefixes:
            if ap == pref or ap.startswith(pref + os.sep):
                return True
        return False

    for root, _, files in os.walk(prior_dir):
        for file_name in files:
            lower_name = file_name.lower()
            if lower_name.endswith((".pt", ".jpg", ".jpeg", ".png")):
                full = os.path.join(root, file_name)
                if not _skip(full):
                    candidate_paths.append(full)

    if not candidate_paths:
        raise ValueError(f"No face files found in {prior_dir} (after exclusions)")

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


def _download_with_timeout(url: str, dest_path: str, timeout_s: int = STYLEGAN_DOWNLOAD_TIMEOUT_S) -> None:
    """Download a file with an explicit timeout and a temporary file for atomic replacement."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = f"{dest_path}.tmp"

    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response, open(temp_path, "wb") as handle:
            chunk_size = 1024 * 1024
            content_length = response.headers.get("Content-Length")
            total_bytes = int(content_length) if content_length and content_length.isdigit() else None
            downloaded = 0

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    percent = (100.0 * downloaded) / total_bytes
                    print(f"      download progress: {percent:5.1f}%", end="\r", flush=True)

        if total_bytes:
            print(" " * 60, end="\r", flush=True)
    except Exception as exc:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        raise RuntimeError(
            f"Failed to download {url} within {timeout_s}s. Check internet, proxy, or firewall settings."
        ) from exc

    os.replace(temp_path, dest_path)


def _ensure_stylegan_checkpoint(network_pkl: str, timeout_s: int = STYLEGAN_DOWNLOAD_TIMEOUT_S) -> str:
    """Ensure the StyleGAN checkpoint is present locally and return the file path used by dnnlib."""
    if not str(network_pkl).lower().startswith(("http://", "https://")):
        return network_pkl

    cache_root = os.path.join(tempfile.gettempdir(), "stylegan2-ada-pytorch-cache", "checkpoints")
    os.makedirs(cache_root, exist_ok=True)

    url_hash = hashlib.md5(network_pkl.encode("utf-8")).hexdigest()
    ext = os.path.splitext(network_pkl)[1] or ".pkl"
    cached_path = os.path.join(cache_root, f"{url_hash}{ext}")

    if os.path.exists(cached_path):
        return cached_path

    print(f"    Downloading StyleGAN checkpoint to cache: {cached_path}")
    _download_with_timeout(network_pkl, cached_path, timeout_s=timeout_s)
    return cached_path


def load_stylegan_generator(network_pkl: str, stylegan_repo_dir: Optional[str] = None, device: str = "cpu"):
    """Load a pretrained StyleGAN2/StyleGAN3 generator from an official NVLabs pickle."""
    print("    [StyleGAN] Preparing generator...")
    repo_dir = stylegan_repo_dir
    if not repo_dir or not os.path.exists(repo_dir):
        print("    [StyleGAN] Ensuring repo cache...")
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

    checkpoint_path = _ensure_stylegan_checkpoint(network_pkl)
    print(f"    [StyleGAN] Loading checkpoint: {checkpoint_path}")
    with dnnlib.util.open_url(checkpoint_path) as fp:
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
        print(f"    Downloading StyleGAN repo archive: {STYLEGAN_REPO_URL}")
        _download_with_timeout(STYLEGAN_REPO_URL, archive_path)

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
    latent_reg_weight: float = 0.001,
    save_path: str = None,
    seed: int = None,
):
    """
    StyleGAN-based inversion in W space — embedding objective only (attacker never sees victim pixels).
    """
    device = next(generator.parameters()).device
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    target_embedding = target_embedding.to(device)

    if seed is not None:
        torch.manual_seed(seed)

    z_dim = int(getattr(generator, "z_dim", 512))
    class_labels = None
    if hasattr(generator, "c_dim") and getattr(generator, "c_dim", 0) > 0:
        class_labels = torch.zeros([1, generator.c_dim], device=device)

    # Start from mapped W space (much more expressive than optimizing Z alone).
    with torch.no_grad():
        z0 = torch.randn(1, z_dim, device=device)
        w0 = generator.mapping(z0, class_labels)

    latent = w0.detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_history = []
    with torch.no_grad():
        try:
            start_full = _stylegan_synthesize(generator, latent)
            start_small = _downsample_for_facenet(start_full)
            start_emb = model(start_small)
            initial_dist = float(torch.norm(start_emb - target_embedding).item())
        except Exception:
            initial_dist = None

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(iterations // 3, 1), gamma=0.5)

    for i in range(iterations):
        optimizer.zero_grad()

        synth_full = _stylegan_synthesize(generator, latent)
        synth_160 = _downsample_for_facenet(synth_full)
        generated_emb = model(synth_160)

        identity_loss = _embedding_inversion_loss(generated_emb, target_embedding)

        latent_reg = torch.mean(latent.pow(2))

        loss = identity_weight * identity_loss + latent_reg_weight * latent_reg

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(float(loss.item()))

        if i % 200 == 0 or i == iterations - 1:
            print(
                f"    iter {i:4d}: identity={identity_loss.item():.6f}, "
                f"latent_reg={latent_reg.item():.6f}, total={loss.item():.6f}"
            )

    with torch.no_grad():
        final_full = _stylegan_synthesize(generator, latent).detach()
        final_small = _downsample_for_facenet(final_full).detach()

    if save_path:
        try:
            from torchvision.utils import save_image
            img_save = (final_small.squeeze() + 1.0) / 2.0
            save_image(img_save, save_path)
        except Exception:
            pass

    return final_small, loss.item(), {"initial_distance": initial_dist, "loss_history": loss_history}

def run_inversion_attack(
    model,
    target_embedding: torch.Tensor,
    iterations: int = 2000,
    lr: float = 0.01,
    start_tensor: torch.Tensor = None,
    prior_dir: Optional[str] = None,
    prior_exclude_prefixes: Optional[list] = None,
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

    # Face-shaped prior beats pure noise under high-d embedding loss (still not the ground-truth crop).
    if start_tensor is not None:
        fake_img = start_tensor.clone().float()
    elif prior_dir and os.path.isdir(prior_dir):
        try:
            fake_img = load_face_prior(
                prior_dir,
                max_images=64,
                exclude_prefixes=prior_exclude_prefixes or [],
            ).detach().clone()
        except Exception:
            fake_img = torch.randn(1, 3, FACENET_INPUT, FACENET_INPUT) * 0.15
    else:
        fake_img = torch.randn(1, 3, FACENET_INPUT, FACENET_INPUT) * 0.15

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

    tgt = target_embedding.detach()
    for i in range(iterations):
        optimizer.zero_grad()
        fake_emb = model(fake_img)

        cos_loss = _embedding_inversion_loss(fake_emb, tgt)
        mse = criterion(fake_emb, tgt)
        # Cosine dominates (stable for embeddings); small MSE nudges scale.
        match_loss = cos_loss + 0.05 * mse

        tv = (
            torch.mean(torch.abs(fake_img[:, :, :, :-1] - fake_img[:, :, :, 1:])) +
            torch.mean(torch.abs(fake_img[:, :, :-1, :] - fake_img[:, :, 1:, :]))
        )

        loss = match_loss + 5e-5 * tv

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            fake_img.clamp_(-1.0, 1.0)

        loss_history.append(float(loss.item()))

        if i % 200 == 0 or i == iterations - 1:
            print(
                f"    iter {i:4d}: cos={(cos_loss.item()):.6f}, mse={mse.item():.6f}, "
                f"tv={tv.item():.6f}, total={loss.item():.6f}"
            )

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
    iterations: int = 1000,
    attack_lr: float = 0.02,
    stylegan_network_pkl: Optional[str] = None,
    stylegan_repo_dir: Optional[str] = None,
    face_prior_dir: Optional[str] = None,
    random_seed: int = 42,
    plot_file_tag: Optional[str] = None,
) -> dict:
    """
    Inversion on BOTH models. Optimiser sees only embeddings (white-box embedding leak).
    Ground-truth crop is loaded afterward for evaluator plots — never fed into losses.
    """
    os.makedirs(output_dir, exist_ok=True)

    tag = (plot_file_tag or os.path.basename(os.path.abspath(client_dir))).strip() or "client"
    vic_abs = os.path.abspath(client_dir)

    stylegan_ckpt = (stylegan_network_pkl or "").strip() or (STYLEGAN_NETWORK_PKL or "").strip()
    repo_dir = stylegan_repo_dir if stylegan_repo_dir else STYLEGAN_REPO_DIR
    repo_dir_effective = repo_dir.strip() if isinstance(repo_dir, str) and repo_dir.strip() else None

    infer_device = "cuda" if torch.cuda.is_available() else "cpu"

    prior = face_prior_dir
    if prior is None and os.path.isdir(CROPPED_DIR):
        prior = CROPPED_DIR

    print(
        "  [Attack] Embedding-only inversion (pixel-space; no victim pixels in loss)."
    )

    def run_one_attack(
        model,
        embedding,
        caption: str,
        out_png: str,
        prior_exclude_prefixes: list,
    ) -> tuple[torch.Tensor, float]:
        # StyleGAN disabled: use pixel-space inversion only
        print(f"    {caption}: pixel-space inversion ({iterations} iters, lr={attack_lr})")
        img, loss_val = run_inversion_attack(
            model,
            embedding,
            iterations=iterations,
            lr=attack_lr,
            prior_dir=prior if prior else None,
            prior_exclude_prefixes=prior_exclude_prefixes,
            save_path=out_png,
            save_loss_path=None,
            seed=random_seed,
        )
        return img, loss_val

    print("  Running on Version A (no DP)...")
    model_no_dp = load_model_from_checkpoint(model_no_dp_path)
    target_emb_no_dp, sid_name = whitebox_target_embedding_from_crop(model_no_dp, client_dir)

    prior_exclude_roots = [vic_abs]
    if os.path.isdir(CROPPED_DIR) and sid_name:
        crop_abs = os.path.abspath(os.path.join(CROPPED_DIR, sid_name))
        if os.path.isdir(crop_abs):
            prior_exclude_roots.append(crop_abs)

    path_no_dp = os.path.join(output_dir, f"attack_no_dp_{tag}.png")
    fake_img_no_dp, final_loss_no_dp = run_one_attack(
        model_no_dp, target_emb_no_dp, "No DP", path_no_dp, prior_exclude_roots
    )
    print(f"  Saved: {path_no_dp}")

    print("  Running on Version B (with DP)...")
    model_with_dp = load_model_from_checkpoint(model_with_dp_path)
    target_emb_with_dp, _ = whitebox_target_embedding_from_crop(model_with_dp, client_dir)

    path_with_dp = os.path.join(output_dir, f"attack_with_dp_{tag}.png")
    fake_img_with_dp, final_loss_with_dp = run_one_attack(
        model_with_dp, target_emb_with_dp, "With DP", path_with_dp, prior_exclude_roots
    )
    print(f"  Saved: {path_with_dp}")

    gt_tensor, plot_name = evaluator_ground_truth_crop(client_dir)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Model inversion — {tag}", fontsize=12, y=1.02)

    gt_img = ((gt_tensor.squeeze() + 1) / 2).clamp(0, 1)
    axes[0].imshow(gt_img.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title(
        f"Original (victim crop)\n{plot_name} — not used in loss"
    )
    axes[0].axis("off")

    attack_no_dp = ((fake_img_no_dp.squeeze() + 1) / 2).clamp(0, 1)
    axes[1].imshow(attack_no_dp.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title("Inversion without DP", color="darkred")
    axes[1].axis("off")

    attack_with_dp = ((fake_img_with_dp.squeeze() + 1) / 2).clamp(0, 1)
    axes[2].imshow(attack_with_dp.permute(1, 2, 0).cpu().numpy())
    axes[2].set_title("Inversion with DP", color="darkgreen")
    axes[2].axis("off")

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, f"attack_comparison_{tag}.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {comparison_path}")

    return {
        "client_tag": tag,
        "no_dp_final_loss": final_loss_no_dp,
        "with_dp_final_loss": final_loss_with_dp,
    }
