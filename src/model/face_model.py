from facenet_pytorch import InceptionResNetV1
import torch
import numpy as np

def get_model(mode="train"):
    # We load the pretrained InceptionResNetV1 face model.
    model = InceptionResNetV1(pretrained='vggface2')
    
    # Just set it to train or eval depending on what we need
    if mode == "train":
        model.train()
    else:
        model.eval()
        
    return model

def get_parameters(model):
    # Flower framework only speaks NumPy, so we have to convert all of PyTorch's weights 
    # (state_dict) into a simple list of NumPy arrays.
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    # Reverse of the above: the server sends us NumPy arrays, and we have to zip them 
    # back into the PyTorch model's dictionary keys.
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)