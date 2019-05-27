import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'run on device: {device}')
