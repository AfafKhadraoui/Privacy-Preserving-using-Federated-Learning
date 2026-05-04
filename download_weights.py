import os
os.environ['TORCH_HOME'] = r'F:\cns_project_cache\torch'
print('TORCH_HOME =', os.environ['TORCH_HOME'])
from facenet_pytorch import InceptionResnetV1
print('Downloading facenet vggface2 weights...')
InceptionResnetV1(pretrained='vggface2')
print('Download complete')
