import sys
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.face_model import (
    get_model,
    get_parameters,
    set_parameters,
    federated_averaging,
    save_model,
    load_model
)


def test_model_forward():
    print("\n[TEST] Model forward pass")

    model = get_model()
    model.eval()  # FIX: disable training mode (BatchNorm issue)

    dummy_input = torch.randn(1, 3, 160, 160)  # FaceNet input size

    with torch.no_grad():  # FIX: no gradient needed for testing
        output = model(dummy_input)

    print("Output shape:", output.shape)
    assert output.shape[1] == 512  # InceptionResnetV1 default embedding size

    print("[OK] Forward pass works")


def test_save_load():
    print("\n[TEST] Save / Load model")

    model = get_model()

    path = "tests/temp_model.pth"

    save_model(model, path)
    loaded_model = load_model(path)

    assert loaded_model is not None

    print("[OK] Save and load works")


def test_fl_parameters():
    print("\n[TEST] FL parameter conversion")

    model = get_model()

    params = get_parameters(model)

    assert isinstance(params, list)
    assert len(params) > 0

    new_model = get_model()
    new_model = set_parameters(new_model, params)

    print("[OK] get/set parameters works")


def test_fedavg():
    print("\n[TEST] Federated Averaging")

    model = get_model()

    params1 = get_parameters(model)
    params2 = get_parameters(model)
    params3 = get_parameters(model)

    aggregated = federated_averaging(
        [params1, params2, params3],
        client_sizes=[10, 20, 30]
    )

    assert len(aggregated) == len(params1)

    print("[OK] FedAvg works")


if __name__ == "__main__":
    test_model_forward()
    test_save_load()
    test_fl_parameters()
    test_fedavg()

    print("\nALL TESTS PASSED")