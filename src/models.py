import torch.nn as nn

class AnomalyTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=4, num_layers=2, num_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)          # [B, L, d_model]
        h = self.encoder(x)             # [B, L, d_model]
        h = h.mean(dim=1)               # Pool over time
        out = self.classifier(h)        # [B, num_classes]
        return out

