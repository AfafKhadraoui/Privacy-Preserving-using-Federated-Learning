"""
dataset_pipeline_validator.py
Project: CNS-project (Federated Face Recognition)

Purpose:
1. Scan raw dataset (data/raw)
2. Run preprocessing (face detection + cropping)
3. Scan cropped dataset (data/cropped)
4. Compare both and generate a per-image status report

Output:
- Dictionary report per user per image
"""

from pathlib import Path
from src.preprocessing.prepare_dataset import run_preprocessing


# ----------------------------
# CONFIG
# ----------------------------
RAW_DIR = Path("data/raw")
CROPPED_DIR = Path("data/cropped")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ----------------------------
# STEP 1: SCAN RAW DATA
# ----------------------------
def scan_raw_dataset(raw_dir: Path):
    dataset = {}

    for user_dir in raw_dir.iterdir():
        if not user_dir.is_dir():
            continue

        images = [
            f.name for f in user_dir.iterdir()
            if f.suffix.lower() in VALID_EXTENSIONS
        ]

        dataset[user_dir.name] = images

    return dataset


# ----------------------------
# STEP 2: SCAN CROPPED DATA
# ----------------------------
def scan_cropped_dataset(cropped_dir: Path):
    dataset = {}

    if not cropped_dir.exists():
        return dataset

    for user_dir in cropped_dir.iterdir():
        if not user_dir.is_dir():
            continue

        dataset[user_dir.name] = {
            f.stem + f.suffix: f.name
            for f in user_dir.iterdir()
            if f.suffix == ".pt"
        }

    return dataset


# ----------------------------
# STEP 3: BUILD COMPARISON REPORT
# ----------------------------
def build_report(raw, cropped):
    report = {}

    for user, images in raw.items():
        report[user] = {}

        for img in images:
            pt_name = Path(img).stem + ".pt"

            if user not in cropped:
                report[user][img] = "USER_MISSING_IN_CROPPED"
            elif pt_name in cropped[user]:
                report[user][img] = "OK"
            else:
                report[user][img] = "SKIPPED_OR_FAILED"

    return report


# ----------------------------
# STEP 4: RUN FULL PIPELINE
# ----------------------------
def run_pipeline():
    print("\n[1] Scanning raw dataset...")
    raw = scan_raw_dataset(RAW_DIR)

    print("[2] Running preprocessing (face detection + cropping)...")
    run_preprocessing()

    print("[3] Scanning cropped dataset...")
    cropped = scan_cropped_dataset(CROPPED_DIR)

    print("[4] Building validation report...\n")
    report = build_report(raw, cropped)

    return report


# ----------------------------
# OPTIONAL: pretty print report
# ----------------------------
def print_report(report):
    for user, images in report.items():
        print(f"\nUser: {user}")
        for img, status in images.items():
            print(f"  {img} -> {status}")


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    report = run_pipeline()
    print_report(report)