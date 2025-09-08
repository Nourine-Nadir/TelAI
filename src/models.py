import torch.nn as nn

class AnomalyTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=8, num_layers=12, num_classes=2):
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


import torch
import math


class AnomalyAwareTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=8, num_layers=6, num_classes=2, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Multi-scale feature extraction
        self.attention_pool = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)  # [B, L, d_model]
        x = self.pos_encoding(x)  # Add positional encoding

        # Transformer encoding
        h = self.encoder(x)  # [B, L, d_model]

        # Anomaly attention scoring
        anomaly_scores, _ = self.attention_pool(h, h, h)  # [B, L, d_model]
        anomaly_weights = self.anomaly_scorer(anomaly_scores)  # [B, L, 1]

        # Weighted pooling
        weighted_features = h * anomaly_weights  # [B, L, d_model]
        pooled_features = weighted_features.mean(dim=1)  # [B, d_model]

        # Also keep original mean features
        mean_features = h.mean(dim=1)  # [B, d_model]

        # Concatenate both representations
        combined = torch.cat([pooled_features, mean_features], dim=1)  # [B, d_model*2]

        # Classification
        out = self.classifier(combined)  # [B, num_classes]
        return out, anomaly_weights.squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)