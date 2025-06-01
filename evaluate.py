#%%
# evaluate.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from models.efficientnet_embedder import EfficientNetEmbedder
from utils.dataset import Market1501Dataset
import config

# Transforms 
transform = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load Gallery and Query Sets 
gallery_set = Market1501Dataset(root=config.DATA_DIR + "bounding_box_test", transform=transform)
query_set = Market1501Dataset(root=config.DATA_DIR + "query", transform=transform)

gallery_loader = DataLoader(gallery_set, batch_size=config.BATCH_SIZE, shuffle=False)
query_loader = DataLoader(query_set, batch_size=1, shuffle=False)

# Load Model 
model = EfficientNetEmbedder(embed_dim=config.EMBED_DIM).to(config.DEVICE)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Extract gallery embeddings
def extract_embeddings(loader):
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting Embeddings"):
            imgs = imgs.to(config.DEVICE)
            embs = model(imgs).cpu().numpy()
            all_embeddings.append(embs)
            all_labels.extend(labels.numpy())
    return np.vstack(all_embeddings), np.array(all_labels)

gallery_embeddings, gallery_labels = extract_embeddings(gallery_loader)
query_embeddings, query_labels = extract_embeddings(query_loader)

# Evaluation Metrics 
def evaluate_rank_map(query_embs, query_labels, gallery_embs, gallery_labels, top_k=(1, 5, 10)):
    sims = cosine_similarity(query_embs, gallery_embs)
    num_queries = query_embs.shape[0]

    rank_k_hits = {k: 0 for k in top_k}
    average_precisions = []

    for i in tqdm(range(num_queries), desc="Evaluating Rank-k & mAP"):
        sim_scores = sims[i]
        sorted_indices = np.argsort(sim_scores)[::-1]  # Descending
        true_label = query_labels[i]
        matches = (gallery_labels[sorted_indices] == true_label).astype(int)

        # Rank-k
        for k in top_k:
            if np.any(matches[:k]):
                rank_k_hits[k] += 1

        # AP
        cum_hits = np.cumsum(matches)
        precision_at_i = cum_hits / (np.arange(1, len(matches)+1))
        ap = np.sum(precision_at_i * matches) / max(np.sum(matches), 1)
        average_precisions.append(ap)

    # Accuracy and mAP
    rank_k_acc = {f"Rank-{k}": rank_k_hits[k] / num_queries for k in top_k}
    mean_ap = np.mean(average_precisions)

    return rank_k_acc, mean_ap

# Run Evaluation 
rank_acc, mean_ap = evaluate_rank_map(query_embeddings, query_labels, gallery_embeddings, gallery_labels)

# Print Results 
print("\nEvaluation Results:")
for k, v in rank_acc.items():
    print(f"{k} Accuracy: {v*100:.2f}%")
print(f"Mean Average Precision (mAP): {mean_ap*100:.2f}%")

# %%
