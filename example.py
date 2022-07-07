"""
Run the HMAX model on the example images.

Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle

import hmax

# Initialize the model with the universal patch set
print('Constructing model')
model = hmax.HMAX('./universal_patch_set.mat')

# A folder with example images
example_images = datasets.ImageFolder(
    './example_images/',
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])
)

# A dataloader that will run through all example images in one batch
dataloader = DataLoader(example_images, batch_size=10)

# Determine whether there is a compatible GPU available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Run the model on the example images
print('Running model on', device)
model = model.to(device)
for X, y in dataloader:
    s1, c1, s2, c2 = model.get_all_layers(X.to(device))

print('Saving output of all layers to: output.pkl')
with open('output.pkl', 'wb') as f:
    pickle.dump(dict(s1=s1, c1=c1, s2=s2, c2=c2), f)
print('[done]')
