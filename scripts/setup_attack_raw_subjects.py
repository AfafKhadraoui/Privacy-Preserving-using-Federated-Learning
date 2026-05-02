"""
Prepare two folders under data/raw/ for preprocessing into client_00 and client_01.

Usage (from repo root):
  python scripts/setup_attack_raw_subjects.py

If data/raw/try.jpg exists:
  copies to data/raw/subject_try/
  saves a horizontally flipped image to data/raw/subject_flip/face.jpg (second raw file for the pipeline).
Otherwise prints what to drop in each folder manually.
"""
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SRC_TRY = RAW / "try.jpg"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    a = RAW / "subject_try"
    b = RAW / "subject_flip"
    a.mkdir(parents=True, exist_ok=True)
    b.mkdir(parents=True, exist_ok=True)

    if SRC_TRY.is_file():
        import shutil

        shutil.copy(SRC_TRY, a / "face.jpg")
        print(f"Copied {SRC_TRY} -> {a / 'face.jpg'}")
        if Image is None:
            print("Install Pillow to auto-create subject_flip: pip install Pillow")
            return
        img = Image.open(SRC_TRY).convert("RGB")
        img.transpose(Image.FLIP_LEFT_RIGHT).save(b / "face.jpg")
        print(f"Saved flipped duplicate -> {b / 'face.jpg'} (replace with another person for a real AB test.)")
        print("\nNext: run your preprocessing so client_00 / client_01 map to two identities, e.g.")
        print("  python src/preprocessing/prepare_dataset.py")
        return

    print("Place your first photo at: data/raw/try.jpg")
    print("Then re-run this script, or manually create:")
    print(f"  {a}/<any>.jpg   -> map to client_00 after preprocessing")
    print(f"  {b}/<any>.jpg   -> second subject -> client_01 after preprocessing")


if __name__ == "__main__":
    main()
