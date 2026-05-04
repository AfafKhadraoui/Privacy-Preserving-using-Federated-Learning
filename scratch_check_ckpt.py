import torch
import os

repo_root = r"f:\summer\S6\Security\cns\CNS-project"
ckpt_path = os.path.join(repo_root, "cns_project_cache", "downloads", "mobilestylegan_ffhq.ckpt")

if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print("Keys in checkpoint:", ckpt.keys())
    if "state_dict" in ckpt:
        print("First 10 state_dict keys:", list(ckpt["state_dict"].keys())[:10])
    if "params" in ckpt:
        print("Params:", ckpt["params"])
else:
    print("Checkpoint not found at", ckpt_path)
