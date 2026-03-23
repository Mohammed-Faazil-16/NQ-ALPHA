import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class NeuroQuantAlphaModel(nn.Module):
    def __init__(self, feature_dim, model_type="transformer"):
        super().__init__()
        self.model_type = model_type.lower()
        self.hidden_dim = 96
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )

        if self.model_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=4,
                dim_feedforward=192,
                dropout=0.15,
                batch_first=True,
                activation="gelu",
                norm_first=False,
            )
            self.position_encoding = PositionalEncoding(self.hidden_dim)
            self.temporal_model = nn.TransformerEncoder(encoder_layer, num_layers=2)
        elif self.model_type == "lstm":
            self.position_encoding = None
            self.temporal_model = nn.LSTM(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=2,
                dropout=0.15,
                batch_first=True,
            )
        else:
            raise ValueError("model_type must be 'transformer' or 'lstm'")

        self.regime_embedding = nn.Embedding(4, 16)
        self.alpha_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 16, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, features, regime_ids):
        if features.dim() == 4:
            batch_size, num_assets, seq_len, feat_dim = features.shape
            flat_features = features.reshape(batch_size * num_assets, seq_len, feat_dim)
            flat_regimes = regime_ids.reshape(batch_size * num_assets)
            flat_scores = self._forward_flat(flat_features, flat_regimes)
            return flat_scores.reshape(batch_size, num_assets)

        if features.dim() == 3:
            return self._forward_flat(features, regime_ids)

        raise ValueError("features must have shape (batch, seq, feat) or (batch, assets, seq, feat)")

    def _forward_flat(self, features, regime_ids):
        regime_ids = regime_ids.long().to(features.device)
        if regime_ids.dim() > 1:
            regime_ids = regime_ids.squeeze(-1)
        regime_ids = torch.clamp(regime_ids, 0, 3)

        batch_size, seq_len, feat_dim = features.shape
        x = features.reshape(batch_size * seq_len, feat_dim)
        encoded = self.feature_projection(x).reshape(batch_size, seq_len, self.hidden_dim)

        if self.model_type == "transformer":
            temporal_output = self.temporal_model(self.position_encoding(encoded))
            temporal_state = temporal_output[:, -1, :]
        else:
            temporal_output, _ = self.temporal_model(encoded)
            temporal_state = temporal_output[:, -1, :]

        regime_state = self.regime_embedding(regime_ids)
        combined = torch.cat([temporal_state, regime_state], dim=1)
        return self.alpha_head(combined).squeeze(-1)
