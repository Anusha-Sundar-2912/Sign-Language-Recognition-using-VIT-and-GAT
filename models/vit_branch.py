import torch
import torch.nn as nn
import timm


class ViTBranch(nn.Module):

    def __init__(self):

        super().__init__()

        # Load pretrained ViT
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True
        )

        # Freeze backbone for faster training
        for param in self.vit.parameters():
            param.requires_grad = False

        # Remove classification head
        self.vit.reset_classifier(0)

        # Temporal attention across frames
        self.temporal_attention = nn.Sequential(
            nn.Linear(768, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Projection for fusion with GAT features
        self.proj = nn.Linear(768, 512)

    def forward(self, x):

        # Input shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # Merge batch and time
        x = x.view(B * T, C, H, W)

        # Extract features from ViT
        features = self.vit(x)  # [B*T, 768]

        # Restore time dimension
        features = features.view(B, T, -1)  # [B, T, 768]

        # Temporal attention weights
        weights = self.temporal_attention(features)
        weights = torch.softmax(weights, dim=1)

        # Weighted temporal pooling
        features = (features * weights).sum(dim=1)  # [B, 768]

        # Projection layer
        features = self.proj(features)  # [B, 512]

        return features