# utils/triplet_loss.py
import torch
import torch.nn.functional as F

def pairwise_distance(embeddings):
    dot = torch.matmul(embeddings, embeddings.t())
    square = dot.diag().unsqueeze(1)
    distances = square - 2 * dot + square.t()
    return torch.clamp(distances, min=0.0)

def hard_triplet_mining(embeddings, labels, margin):
    distances = pairwise_distance(embeddings)
    B = labels.size(0)
    
    # ✅ Initialize loss as a Tensor on the correct device
    loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    for i in range(B):
        anchor = embeddings[i]
        label = labels[i]
        positive_mask = labels == label
        negative_mask = labels != label

        pos_dists = distances[i][positive_mask]
        neg_dists = distances[i][negative_mask]

        if len(pos_dists) > 1 and len(neg_dists) > 0:
            hardest_pos = pos_dists.max()
            hardest_neg = neg_dists.min()
            loss = loss + F.relu(hardest_pos - hardest_neg + margin)  # ✅ still a tensor

    return loss / B  # ✅ returns a tensor