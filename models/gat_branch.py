import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GATBranch(nn.Module):

    def __init__(self):
        super().__init__()

        self.gat1 = GATConv(3, 64, heads=4)
        self.gat2 = GATConv(256, 128)

        # MediaPipe hand skeleton
        self.edge_index = torch.tensor([
            [0,1,2,3,4, 0,5,6,7,8, 0,9,10,11,12, 0,13,14,15,16, 0,17,18,19,20],
            [1,2,3,4,0, 5,6,7,8,0, 9,10,11,12,0, 13,14,15,16,0, 17,18,19,20,0]
        ], dtype=torch.long)

    def forward(self, keypoints):

        # keypoints: [B, T, 21, 3]
        B, T, J, C = keypoints.shape
        device = keypoints.device

        edge_index = self.edge_index.to(device)

        # flatten batch and time
        x = keypoints.view(B * T * J, C)

        # create graph index offsets
        edge_index = edge_index.repeat(1, B*T)
        offset = torch.arange(B*T, device=device).repeat_interleave(edge_index.shape[1]//(B*T)) * J
        edge_index = edge_index + offset

        x = self.gat1(x, edge_index)
        x = torch.relu(x)

        x = self.gat2(x, edge_index)
        x = torch.relu(x)

        x = x.view(B*T, J, 128)

        # graph pooling
        x = x.mean(dim=1)

        # temporal pooling
        x = x.view(B, T, 128)
        x = x.mean(dim=1)

        return x