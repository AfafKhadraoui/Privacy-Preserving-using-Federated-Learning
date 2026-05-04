import os, torch, inspect
import facenet_pytorch
from facenet_pytorch.models import inception_resnet_v1

print('ENV_TORCH_HOME =', os.environ.get('TORCH_HOME'))
print('TORCH_HUB_DIR =', torch.hub.get_dir())
print('facenet_pytorch __file__ =', facenet_pytorch.__file__)
print('\n--- load_weights source (first 80 lines) ---')
try:
    src = inspect.getsource(inception_resnet_v1.load_weights)
    print('\n'.join(src.splitlines()[:80]))
except Exception as e:
    print('Error getting source:', e)

# Also print common cache locations
home = os.path.expanduser('~')
print('\nUser home:', home)
print('Possible cache paths:')
print(os.path.join(home, '.cache', 'torch', 'checkpoints'))
print(os.path.join(home, '.cache', 'torch'))
print(torch.hub.get_dir())
print('TORCH_HOME env var present:', os.environ.get('TORCH_HOME') is not None)
