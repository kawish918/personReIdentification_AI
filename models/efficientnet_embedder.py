import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetEmbedder(nn.Module):
    def __init__(self, embed_dim=256):
        super(EfficientNetEmbedder, self).__init__()
        base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(base_model.classifier[1].in_features, embed_dim)

    def forward(self, x):
        x = self.features(x)                
        x = self.pool(x)                    
        x = x.view(x.size(0), -1)           
        x = self.embedding(x)              
        return x