#%%
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
from models.efficientnet_embedder import EfficientNetEmbedder


# === Configuration ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_model.pth'
QUERY_IMAGE_PATH = 'C:/Users/kawis/Desktop/Py/Person_Re_Id/test_images/p10.jpg'
GALLERY_DIR = 'test_images/'  
TOP_K = 4

# === Image Transform (same as used during training) ===
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Load Model ===


model = EfficientNetEmbedder(embed_dim=512).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# === Feature Extraction ===
def extract_feature(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(image)
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.squeeze(0)  # [512]

# === Load Query and Gallery Features ===
query_feat = extract_feature(QUERY_IMAGE_PATH)

gallery_feats = []
gallery_paths = []

for fname in os.listdir(GALLERY_DIR):
    path = os.path.join(GALLERY_DIR, fname)
    feat = extract_feature(path)
    gallery_feats.append(feat)
    gallery_paths.append(path)

gallery_feats = torch.stack(gallery_feats)  # [N, 512]

# === Compute Similarities ===
similarities = F.cosine_similarity(query_feat.unsqueeze(0), gallery_feats)

# === Top-K Matches ===
top_k_indices = similarities.topk(TOP_K).indices.cpu().numpy()

# === Display ===
fig, axes = plt.subplots(1, TOP_K + 1, figsize=(15, 5))

# Query
query_img = Image.open(QUERY_IMAGE_PATH).convert('RGB')
axes[0].imshow(query_img)
axes[0].set_title('Query')
axes[0].axis('off')

# Top-K Gallery Matches
for i, idx in enumerate(top_k_indices):
    match_path = gallery_paths[idx]
    match_img = Image.open(match_path).convert('RGB')
    axes[i + 1].imshow(match_img)
    axes[i + 1].set_title(f'Top {i+1}\n{os.path.basename(match_path)}')
    axes[i + 1].axis('off')

plt.tight_layout()
plt.savefig('output.jpg')
plt.show()

# %%
