# Person Re-Identification using EfficientNet and Triplet Loss

This project performs **Person Re-Identification (Re-ID)** using deep feature embeddings extracted via a **pretrained EfficientNet-B0** model. The system retrieves visually similar identities from a gallery using **cosine similarity** on learned embeddings.

---

## 📁 Dataset

We used the **[Market-1501 dataset](https://github.com/zhunzhong07/Market-1501)** which consists of:
- 751 training identities
- 750 test identities
- Images captured from 6 camera views

---

## 🧠 Model Details

- **Backbone**: EfficientNet-B0
- **Embedding Size**: 512-dimensional vector
- **Loss Function**: Triplet Loss (using hard negative mining)
- **Similarity Metric**: Cosine similarity
- **Training Epochs**: 50+
- **Framework**: PyTorch

---

## ⚙️ How It Works

1. Each image is passed through a pretrained EfficientNet to extract a **512-d embedding**.
2. A query image is compared to the embeddings of all gallery images using **cosine similarity**.
3. Top-K most similar images are returned as the result.
4. A **Gradio web interface** allows user interaction for live querying.

---
