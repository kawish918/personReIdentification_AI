#%%
import torch
import gradio as gr
from torchvision import transforms
from torch.nn import functional as F
from PIL import Image
import os
import numpy as np
from models.efficientnet_embedder import EfficientNetEmbedder

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_model.pth'
GALLERY_DIR = 'test_images/'
TOP_K = 4

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load model
model = EfficientNetEmbedder(embed_dim=512).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Extract feature
def extract_feature(image):
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(image)
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.squeeze(0)

# Load gallery features once
gallery_paths = []
gallery_feats = []

for fname in os.listdir(GALLERY_DIR):
    path = os.path.join(GALLERY_DIR, fname)
    img = Image.open(path).convert('RGB')
    feat = extract_feature(img)
    gallery_paths.append(path)
    gallery_feats.append(feat)

gallery_feats = torch.stack(gallery_feats)  # [N, 512]

# Inference function
def find_top_k_matches(query_img):
    query_img = query_img.convert("RGB")
    query_feat = extract_feature(query_img)

    similarities = F.cosine_similarity(query_feat.unsqueeze(0), gallery_feats)
    top_k_indices = similarities.topk(TOP_K).indices.cpu().numpy()

    results = [ (query_img, "Query") ]  # First is the query
    for i, idx in enumerate(top_k_indices):
        match_img = Image.open(gallery_paths[idx]).convert("RGB")
        results.append((match_img, f"Top {i+1}"))

    return results


# Gradio Interface
gallery_labels = ["Query"] + [f"Top {i+1}" for i in range(TOP_K)]
gr.Interface(
    fn=find_top_k_matches,
    inputs=gr.Image(type="pil", label="Upload Query Image"),
    outputs=gr.Gallery(label="Results", columns=[TOP_K + 1], height="auto"),
    title="Person Re-Identification Demo",
    description="Upload a person image and see the top-k similar images from the gallery."
).launch()

# %%
