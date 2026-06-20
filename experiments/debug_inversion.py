import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_NO_DP, PLOTS_DIR
# Ensure temp and torch cache are on F: to avoid C: paging issues
os.environ["TORCH_HOME"] = "F:\\cns_project_cache\\torch_home"
os.environ["TEMP"] = "F:\\cns_project_cache\\temp"
os.environ["TMP"] = "F:\\cns_project_cache\\temp"
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
os.makedirs(os.environ["TEMP"], exist_ok=True)

from src.attacks.model_inversion import load_model_from_checkpoint, get_target_embedding, load_face_prior, run_inversion_attack

os.makedirs(PLOTS_DIR, exist_ok=True)

client_dir = "data/clients/client_00"
print("Loading model...")
model = load_model_from_checkpoint(MODEL_NO_DP)

print("Getting target embedding and original image...")
target_emb, person_name, original_tensor, target_path = get_target_embedding(model, client_dir)
print(f"Target from: {target_path} (person: {person_name})")

print("Building face prior from client folder...")
prior = load_face_prior(client_dir, max_images=8)
print("Prior shape:", prior.shape)

save_path = os.path.join(PLOTS_DIR, "debug_attack_no_dp_prior.png")
print("Running inversion with face prior initialization (this may take a while)...")
fake_img, final_loss = run_inversion_attack(
    model,
    target_emb,
    iterations=2000,
    lr=0.01,
    start_tensor=prior,
    save_path=save_path,
    save_loss_path=os.path.join(PLOTS_DIR, "debug_attack_no_dp_loss.json"),
    seed=42,
)

print(f"Saved reconstructed image to: {save_path}")
print(f"Final loss: {final_loss}")
