import torch
import torch.nn.functional as F

def batch_hard_triplet_loss(embeddings, labels, margin):
    # Normalize embeddings for consistency with evaluation
    embeddings = F.normalize(embeddings, p=2, dim=1)
    dist = torch.cdist(embeddings, embeddings, p=2)
    device = embeddings.device

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    valid_triplets = 0

    for i in range(embeddings.size(0)):
        anchor_label = labels[i]
        dists = dist[i]

        positive_mask = labels == anchor_label
        negative_mask = labels != anchor_label
        positive_mask[i] = False  # Remove self from positives

        if positive_mask.any() and negative_mask.any():
            hardest_positive = dists[positive_mask].max()
            hardest_negative = dists[negative_mask].min()
            triplet_loss = F.relu(hardest_positive - hardest_negative + margin)
            total_loss = total_loss + triplet_loss
            valid_triplets += 1
        else:
            print(f"No valid triplet for anchor {i}: Unique labels = {torch.unique(labels)}")

    if valid_triplets > 0:
        total_loss = total_loss / valid_triplets
    else:
        print(f"No valid triplets in batch. Total loss: {total_loss.item()}")

    return total_loss