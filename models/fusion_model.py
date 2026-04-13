import torch
import torch.nn as nn

from models.vit_branch import ViTBranch
from models.gat_branch import GATBranch


class SignLanguageModel(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.vit = ViTBranch()
        self.gat = GATBranch()

        self.gat_proj = nn.Linear(128, 512)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, frames, keypoints):

        v_feat = self.vit(frames)        # [B,512]

        g_feat = self.gat(keypoints)     # [B,128]
        g_feat = self.gat_proj(g_feat)   # [B,512]

        fused = v_feat + g_feat

        out = self.classifier(fused)

        return out