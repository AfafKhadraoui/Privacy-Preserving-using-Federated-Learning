import os
import torch

def check_files():
    corrupted = []
    for root, dirs, files in os.walk("."):
        # Ignore virtual environments
        if "venv" in root or ".venv" in root or "conda-envs" in root:
            continue
            
        for file in files:
            if file.endswith('.pt') or file.endswith('.pth'):
                path = os.path.join(root, file)
                try:
                    torch.load(path, map_location='cpu')
                except Exception as e:
                    print(f"Corrupted file found: {path} - {str(e)}")
                    corrupted.append(path)
                    
    if not corrupted:
        print("No corrupted files found.")
    else:
        print(f"Total corrupted files: {len(corrupted)}")
        print("You should delete these files to fix the error.")

if __name__ == "__main__":
    check_files()
