# utils/dataset.py
import os
import torch
from torchvision import transforms
from PIL import Image

class Market1501Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        for img in sorted(os.listdir(root)):
            if img.endswith(".jpg"):
                pid = int(img.split("_")[0])
                self.samples.append((img, pid))

    def __getitem__(self, idx):
        img_name, pid = self.samples[idx]
        img_path = os.path.join(self.root, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, pid

    def __len__(self):
        return len(self.samples)  