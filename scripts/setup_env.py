import subprocess
import sys


REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "facenet-pytorch",
    "flwr",
    "opencv-python",
    "tqdm",
    "flask",
    "flask-cors"
]


def install_package(package):
    """
    Install a Python package using pip.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def check_and_install():
    """
    Check if required packages are installed.
    If not, install them automatically.
    """

    for package in REQUIRED_PACKAGES:
        try:
            __import__(package.replace("-", "_"))
            print(f"[OK] {package} is already installed")

        except ImportError:
            print(f"[INSTALLING] {package}...")
            install_package(package)
            print(f"[DONE] {package}")


if __name__ == "__main__":
    print("Checking dependencies for CNS Federated Face Recognition Project...\n")
    check_and_install()
    print("\nAll dependencies are ready.")