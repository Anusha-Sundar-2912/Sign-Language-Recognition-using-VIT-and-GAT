import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):

    def __init__(self, embed_dim=512):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        # x shape: [batch, time, features]

        x = self.transformer(x)

        x = x.transpose(1, 2)

        x = self.pool(x)

        x = x.squeeze(-1)

        return x