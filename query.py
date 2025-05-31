#%%
# query.py
import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from models.efficientnet_embedder import EfficientNetEmbedder
from utils.dataset import Market1501Dataset
import config
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

# Configuration
GALLERY_DIR = os.path.join(config.DATA_DIR, "bounding_box_test")
TOP_K = 5
MODEL_PATH = "model_epoch_25.pth"
DEVICE = config.DEVICE

# Load model
model = EfficientNetEmbedder(embed_dim=config.EMBED_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load gallery images and compute their embeddings
gallery_paths = [os.path.join(GALLERY_DIR, img) for img in os.listdir(GALLERY_DIR) if img.endswith('.jpg')]
gallery_embeddings = []
gallery_images = []

print("Computing gallery embeddings...")
with torch.no_grad():
    for img_path in tqdm(gallery_paths):
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        embedding = model(img_tensor)
        gallery_embeddings.append(embedding.squeeze(0))
        gallery_images.append(img_path)

gallery_embeddings = torch.stack(gallery_embeddings)  # [N, embed_dim]

# --- QUERY IMAGE ---
query_path = "C:/Users/kawis/Desktop/Py/Person_Re_Id/data/bounding_box_train/0002_c1s1_069056_02.jpg"  
query_img = Image.open(query_path).convert("RGB")
query_tensor = transform(query_img).unsqueeze(0).to(DEVICE)

# Compute query embedding
with torch.no_grad():
    query_embedding = model(query_tensor)  # [1, embed_dim]

# Compute cosine similarity
similarities = cosine_similarity(query_embedding, gallery_embeddings)  # [N]
top_k_indices = torch.topk(similarities, TOP_K).indices.cpu().numpy()

# --- DISPLAY RESULTS ---
plt.figure(figsize=(15, 5))
plt.subplot(1, TOP_K + 1, 1)
plt.imshow(query_img)
plt.title("Query")
plt.axis("off")

for i, idx in enumerate(top_k_indices):
    img_path = gallery_images[idx]
    img = Image.open(img_path).convert("RGB")
    plt.subplot(1, TOP_K + 1, i + 2)
    plt.imshow(img)
    plt.title(f"Rank {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# %%
