"""
Run the HMAX model on the example images.

Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.io import savemat

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
    output = model(X.to(device))

savemat('./C2_output.mat', dict(C2_output=output))
print('Output of C2 units was saved to: C2_output.mat')

print('[done]')
