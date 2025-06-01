#%%

import os
import torch
import gradio as gr
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.functional import cosine_similarity

from models.efficientnet_embedder import EfficientNetEmbedder
import config

# === Configuration ===
GALLERY_DIR = os.path.join(config.DATA_DIR, "bounding_box_test")
MODEL_PATH = "best_model.pth"
TOP_K = 5
DEVICE = config.DEVICE

# === Image Preprocessing ===
transform = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Load Model ===
model = EfficientNetEmbedder(embed_dim=config.EMBED_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === Prepare Gallery Embeddings ===
gallery_paths = [os.path.join(GALLERY_DIR, f) for f in os.listdir(GALLERY_DIR) if f.endswith(".jpg")]
gallery_embeddings = []

print("Generating gallery embeddings...")
with torch.no_grad():
    for path in gallery_paths:
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        embedding = model(img_tensor).squeeze(0)  # [512]
        gallery_embeddings.append(embedding)

gallery_embeddings = torch.stack(gallery_embeddings).to(DEVICE)  # [N, 512]

# === Inference Function ===
def retrieve_similar_images(query_image):
    query_image = query_image.convert("RGB")
    query_tensor = transform(query_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        query_embedding = model(query_tensor).squeeze(0)  # [512]

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding.unsqueeze(0), gallery_embeddings).squeeze(0)  # [N]
    top_k_indices = similarities.topk(TOP_K).indices.cpu().tolist()

    # Load top-k similar images
    return [Image.open(gallery_paths[i]).convert("RGB") for i in top_k_indices]

# === Gradio Interface ===
iface = gr.Interface(
    fn=retrieve_similar_images,
    inputs=gr.Image(type="pil", label="Upload Query Image"),
    outputs=[gr.Image(type="pil", label=f"Rank {i+1}") for i in range(TOP_K)],
    title="Person Re-Identification",
    description="Upload a query image to find the most visually similar individuals from the gallery dataset using learned embeddings and cosine similarity."
)

iface.launch()

# %%
