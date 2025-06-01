# #%%
# # train.py
# import os
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms
# from tqdm import tqdm
# from models.efficientnet_embedder import EfficientNetEmbedder
# from utils.dataset import Market1501Dataset
# from utils.triplet_loss import hard_triplet_mining
# from torch.cuda.amp import autocast, GradScaler
# import config
# import numpy as np
# import random

# # --- Reproducibility ---
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# # --- Transforms ---
# transform = transforms.Compose([
#     transforms.Resize(config.IMG_SIZE),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# # --- Datasets ---
# dataset = Market1501Dataset(root=config.DATA_DIR + "bounding_box_train", transform=transform)
# val_split = int(0.1 * len(dataset))
# train_split = len(dataset) - val_split
# train_set, val_set = random_split(dataset, [train_split, val_split])
# train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
# val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

# # --- Model, Optimizer, AMP ---
# model = EfficientNetEmbedder(embed_dim=config.EMBED_DIM).to(config.DEVICE)
# optimizer = optim.Adam(model.parameters(), lr=config.LR)
# scaler = GradScaler()
# best_val_loss = float('inf')

# train_losses = []
# val_losses = []
# rank1_accuracies = []
# map_accuracies = []


# # --- Training Loop ---
# for epoch in range(config.EPOCHS):
#     model.train()
#     running_loss = 0.0
#     valid_batches = 0
    
#     for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]"):
#         imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
#         optimizer.zero_grad()

#         with autocast():
#             embeddings = model(imgs)
#             loss = hard_triplet_mining(embeddings, labels, config.MARGIN)

#         if loss.requires_grad and loss.item() > 0 and torch.isfinite(loss):
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             running_loss += loss.item()
#             valid_batches += 1

#     avg_train_loss = running_loss / max(valid_batches, 1)

#     # --- Validation ---
#     model.eval()
#     val_loss = 0.0
#     val_batches = 0
#     with torch.no_grad():
#         for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]"):
#             imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
#             with autocast():
#                 embeddings = model(imgs)
#                 loss = hard_triplet_mining(embeddings, labels, config.MARGIN)
#             if torch.isfinite(loss):
#                 val_loss += loss.item()
#                 val_batches += 1

#     avg_val_loss = val_loss / max(val_batches, 1)
#     print(f"Epoch {epoch+1} Summary: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")

#     # --- Save best model ---
#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         torch.save(model.state_dict(), "best_model.pth")
#         print(f"✅ Best model saved at epoch {epoch+1} with val loss {best_val_loss:.4f}")

# print("Training completed.")

# # %%
# # Train and Validation Loss Plotting
# import matplotlib.pyplot as plt
# def plot_loss(train_losses, val_losses):
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Train Loss', color='blue')
#     plt.plot(val_losses, label='Validation Loss', color='orange')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()
#     plt.grid()
#     plt.savefig("loss_plot.jpg")
#     plt.show()

# #%%
# # Rant 1 accuracy curve
# def plot_rank1_accuracy(rank1_accuracies):
#     plt.figure(figsize=(10, 5))
#     plt.plot(rank1_accuracies, label='Rank-1 Accuracy', color='green')
#     plt.xlabel('Epochs')
#     plt.ylabel('Rank-1 Accuracy (%)')
#     plt.title('Rank-1 Accuracy Over Epochs')
#     plt.legend()
#     plt.grid()
#     plt.savefig("rank1_accuracy_plot.jpg")
#     plt.show()

# #%%
# # mAp curve
# def plot_map_accuracy(map_accuracies):
#     plt.figure(figsize=(10, 5))
#     plt.plot(map_accuracies, label='mAP', color='red')
#     plt.xlabel('Epochs')
#     plt.ylabel('Mean Average Precision (mAP)')
#     plt.title('mAP Over Epochs')
#     plt.legend()
#     plt.grid()
#     plt.savefig("map_accuracy_plot.jpg")
#     plt.show()

#%%
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
import matplotlib.pyplot as plt

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

# Initialize lists to store metrics
train_losses = []
val_losses = []
rank1_accuracies = []
map_accuracies = []

def compute_distance_matrix(query_features, gallery_features):
    """Compute distance matrix between query and gallery features"""
    return torch.cdist(query_features, gallery_features, p=2)

def evaluate_retrieval(query_features, gallery_features, query_labels, gallery_labels):
    """Compute Rank-1 accuracy and mAP"""
    dist_matrix = compute_distance_matrix(query_features, gallery_features)
    
    rank1_correct = 0
    avg_precision_sum = 0
    
    for i in range(len(query_labels)):
        # Get distances for this query
        distances = dist_matrix[i]
        # Sort gallery indices by distance (ascending)
        sorted_indices = torch.argsort(distances)
        
        # Get sorted gallery labels
        sorted_labels = gallery_labels[sorted_indices]
        query_label = query_labels[i]
        
        # Rank-1 accuracy
        if sorted_labels[0] == query_label:
            rank1_correct += 1
        
        # Average Precision (AP)
        relevant_mask = (sorted_labels == query_label)
        if relevant_mask.sum() > 0:
            precision_at_k = torch.cumsum(relevant_mask.float(), dim=0) / torch.arange(1, len(relevant_mask) + 1, device=relevant_mask.device)
            ap = (precision_at_k * relevant_mask.float()).sum() / relevant_mask.sum()
            avg_precision_sum += ap.item()
    
    rank1_accuracy = (rank1_correct / len(query_labels)) * 100
    mean_ap = (avg_precision_sum / len(query_labels)) * 100
    
    return rank1_accuracy, mean_ap

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
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]"):
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
            with autocast():
                embeddings = model(imgs)
                loss = hard_triplet_mining(embeddings, labels, config.MARGIN)
            
            if torch.isfinite(loss):
                val_loss += loss.item()
                val_batches += 1
            
            # Collect embeddings and labels for evaluation
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    avg_val_loss = val_loss / max(val_batches, 1)
    
    # Compute retrieval metrics
    if all_embeddings:
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # For simplicity, use all validation data as both query and gallery
        # In practice, you might want to split them differently
        rank1_acc, mean_ap = evaluate_retrieval(all_embeddings, all_embeddings, all_labels, all_labels)
    else:
        rank1_acc, mean_ap = 0.0, 0.0
    
    # Store metrics
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    rank1_accuracies.append(rank1_acc)
    map_accuracies.append(mean_ap)
    
    print(f"Epoch {epoch+1} Summary: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Rank-1={rank1_acc:.2f}% | mAP={mean_ap:.2f}%")

    # --- Save best model ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"✅ Best model saved at epoch {epoch+1} with val loss {best_val_loss:.4f}")

print("Training completed.")

# %%
# Train and Validation Loss Plotting
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig("loss_plot.jpg")
    plt.show()

#%%
# Rank-1 accuracy curve
def plot_rank1_accuracy(rank1_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(rank1_accuracies, label='Rank-1 Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Rank-1 Accuracy (%)')
    plt.title('Rank-1 Accuracy Over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig("rank1_accuracy_plot.jpg")
    plt.show()

#%%
# mAP curve
def plot_map_accuracy(map_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(map_accuracies, label='mAP', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Average Precision (mAP)')
    plt.title('mAP Over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig("map_accuracy_plot.jpg")
    plt.show()

# %%
# Call plotting functions after training is complete
if train_losses:  # Only plot if we have data
    plot_loss(train_losses, val_losses)
    plot_rank1_accuracy(rank1_accuracies)
    plot_map_accuracy(map_accuracies)
else:
    print("No training data to plot!")
# %%
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# embeddings: a 2D numpy array of shape (N_images, 512)
# labels: actual person IDs

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
reduced = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', alpha=0.7)
plt.colorbar(scatter)
plt.title("t-SNE Visualization of Embeddings")
plt.show()
