import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import os

# Load the dataset with the `large_random_1k` subset
dataset = load_dataset('poloclub/diffusiondb', 'large_random_1k')

# Custom class to apply transforms (PIL to tensor etc)
class TextImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        text = sample['prompt']
        image = sample['image']
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return text, image

# Do we need this?
def collate_fn(batch):
    texts, images = zip(*batch)
    return texts, torch.stack(images)

# Step 4: Create the DataLoader
def get_dataloader(dataset, batch_size=32, shuffle=True):

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    custom_dataset = TextImageDataset(dataset, transform=transform)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader