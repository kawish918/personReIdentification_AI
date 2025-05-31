#%%
# train.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from models.efficientnet_embedder import EfficientNetEmbedder
from utils.dataset import Market1501Dataset
from utils.triplet_loss import hard_triplet_mining
from torch.cuda.amp import autocast, GradScaler
import config
import numpy as np
import random

# --- Reproducibility ---
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Datasets ---
dataset = Market1501Dataset(root=config.DATA_DIR + "bounding_box_train", transform=transform)
val_split = int(0.1 * len(dataset))
train_split = len(dataset) - val_split
train_set, val_set = random_split(dataset, [train_split, val_split])
train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

# --- Model, Optimizer, AMP ---
model = EfficientNetEmbedder(embed_dim=config.EMBED_DIM).to(config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=config.LR)
scaler = GradScaler()
best_val_loss = float('inf')

# --- Training Loop ---
for epoch in range(config.EPOCHS):
    model.train()
    running_loss = 0.0
    valid_batches = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]"):
        imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad()

        with autocast():
            embeddings = model(imgs)
            loss = hard_triplet_mining(embeddings, labels, config.MARGIN)

        if loss.requires_grad and loss.item() > 0 and torch.isfinite(loss):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            valid_batches += 1

    avg_train_loss = running_loss / max(valid_batches, 1)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]"):
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
            with autocast():
                embeddings = model(imgs)
                loss = hard_triplet_mining(embeddings, labels, config.MARGIN)
            if torch.isfinite(loss):
                val_loss += loss.item()
                val_batches += 1

    avg_val_loss = val_loss / max(val_batches, 1)
    print(f"Epoch {epoch+1} Summary: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")

    # --- Save best model ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"✅ Best model saved at epoch {epoch+1} with val loss {best_val_loss:.4f}")

print("🔥 Training completed.")

# %%
