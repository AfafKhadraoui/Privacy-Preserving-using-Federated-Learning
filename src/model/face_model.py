"""
Loads InceptionResnetV1 pretrained model.
Provides get_model(), get_parameters(), set_parameters() 
which are the ONLY functions client.py ever calls.

"""

from facenet_pytorch import InceptionResnetV1   # correct spelling — lowercase 'n' in Resnet
import torch
import numpy as np


def get_model(mode="train"):
    """
    Load pretrained InceptionResnetV1 in embedding mode.
    Outputs 512-dim embedding vectors 
    """
    model = InceptionResnetV1(pretrained='vggface2')

    if mode == "train":
        model.train()
    else:
        model.eval()

    return model


def get_parameters(model):
    """
    Convert model weights to list of numpy arrays for Flower.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    """
    Load numpy arrays back into the model.
    Zips the incoming arrays with the model's state_dict keys.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)