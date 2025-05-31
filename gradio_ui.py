#%%
# gradip_ui.py
import os
import torch
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image
from models.efficientnet_embedder import EfficientNetEmbedder
from utils.dataset import Market1501Dataset
from torch.nn.functional import cosine_similarity
import config

# === Configuration ===
GALLERY_DIR = os.path.join(config.DATA_DIR, "bounding_box_test")
MODEL_PATH = "best_model.pth"
TOP_K = 5
DEVICE = config.DEVICE

# === Model Setup ===
model = EfficientNetEmbedder(embed_dim=config.EMBED_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === Transform ===
transform = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Load Gallery Embeddings ===
gallery_paths = [os.path.join(GALLERY_DIR, f) for f in os.listdir(GALLERY_DIR) if f.endswith(".jpg")]
gallery_images = []
gallery_embeddings = []

print("Preparing gallery embeddings...")
with torch.no_grad():
    for img_path in gallery_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        embedding = model(img_tensor).squeeze(0)
        gallery_embeddings.append(embedding)
        gallery_images.append(img_path)

gallery_embeddings = torch.stack(gallery_embeddings)  # [N, D]


# === Inference Function ===
def retrieve_similar_images(query_img):
    query_img = query_img.convert("RGB")
    query_tensor = transform(query_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        query_embedding = model(query_tensor)

    similarities = cosine_similarity(query_embedding, gallery_embeddings)  # [N]
    top_indices = torch.topk(similarities, TOP_K).indices.cpu().numpy()

    # Collect paths of top-k results
    results = [Image.open(gallery_images[i]).convert("RGB") for i in top_indices]
    return results


# === Gradio UI ===
iface = gr.Interface(
    fn=retrieve_similar_images,
    inputs=gr.Image(type="pil", label="Upload Query Image"),
    outputs=[gr.Image(type="pil", label=f"Rank {i+1}") for i in range(TOP_K)],
    title="Person Re-Identification",
    description="Upload a person image. The system will retrieve the most similar images from the gallery using cosine similarity on learned embeddings."
)

iface.launch()

# %%
