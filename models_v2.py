"""
Improved MAB-Net Architectures
===============================
v2 improvements:
  - MABNetParallel:  Bi-LSTM || Transformer (parallel, not serial)
  - Temporal smoothness loss
  - Spatial coordinate encoding
  - Temporal cyclic encoding (hour-of-day, day-of-year)
  - Configurable window sizes (24/48/72/168h)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Positional & Temporal Encodings
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TemporalEncoding(nn.Module):
    """Cyclic temporal features: hour-of-day + day-of-year."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D), timestamps: (B, T) — hours since epoch or [0..23]."""
        B, T, _ = x.shape
        hour = timestamps.float() % 24
        # Approximate day-of-year from index if timestamps are just indices
        doy = (timestamps.float() / 24.0) % 366  # rough

        hour_sin = torch.sin(2 * math.pi * hour / 24)
        hour_cos = torch.cos(2 * math.pi * hour / 24)
        doy_sin  = torch.sin(2 * math.pi * doy / 366)
        doy_cos  = torch.cos(2 * math.pi * doy / 366)

        t_feats = torch.stack([hour_sin, hour_cos, doy_sin, doy_cos], dim=-1)
        t_enc = self.proj(t_feats)  # (B, T, d_model)
        return x + t_enc


class SpatialEncoding(nn.Module):
    """Encode station latitude & longitude as a learnable spatial feature.

    Coordinates are min-max normalized to [-1, 1] per batch.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor, lat: torch.Tensor, lon: torch.Tensor
                ) -> torch.Tensor:
        """lat, lon: (B,) — station coordinates."""
        B, T, D = x.shape
        # Normalize to [-1, 1] (China lat: 18-54, lon: 73-135)
        lat_norm = (lat - 36) / 18
        lon_norm = (lon - 104) / 31
        spatial = torch.stack([lat_norm, lon_norm], dim=-1)  # (B, 2)
        spatial_enc = self.proj(spatial)                      # (B, d_model)
        spatial_enc = spatial_enc.unsqueeze(1).expand(-1, T, -1)  # (B, T, d_model)
        return x + spatial_enc


# ══════════════════════════════════════════════════════════════════════════════
# Improved Architectures
# ══════════════════════════════════════════════════════════════════════════════

class MABNetParallel(nn.Module):
    """MAB-Net v2: Parallel Bi-LSTM and Transformer branches.

    Key change from v1: Bi-LSTM and Transformer process the input in PARALLEL
    (not serial), then their outputs are fused via learned gating.

    Architecture:
        input → [Bi-LSTM branch] ─┐
        input → [Transformer branch] ─┤─→ Gated Fusion → Output
    """

    def __init__(self, n_feats: int = 3, d_model: int = 128,
                 lstm_hidden: int = 64, lstm_layers: int = 2,
                 n_head: int = 4, n_transformer_layers: int = 2,
                 dropout: float = 0.1,
                 use_spatial: bool = False,
                 use_temporal: bool = False):
        super().__init__()
        self.n_feats = n_feats
        self.d_model = d_model
        self.use_spatial = use_spatial
        self.use_temporal = use_temporal

        # Mask-aware input
        in_dim = 2 * n_feats
        self.input_proj = nn.Linear(in_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)  # v3: stabilize

        # Optional encodings
        if use_temporal:
            self.temporal_enc = TemporalEncoding(d_model)
        if use_spatial:
            self.spatial_enc = SpatialEncoding(d_model)
        else:
            self.pos_enc = SinusoidalPE(d_model)

        # ── Bi-LSTM branch ──
        self.bilstm = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_proj = nn.Linear(lstm_hidden * 2, d_model)
        self.lstm_norm = nn.LayerNorm(d_model)  # v3: stabilize

        # ── Transformer branch ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_transformer_layers)

        # ── Gated fusion ──
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.fusion = nn.Linear(d_model * 2, d_model)

        # ── Output ──
        self.output_proj = nn.Linear(d_model, n_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, observed: torch.Tensor, mask: torch.Tensor,
                lat: torch.Tensor = None, lon: torch.Tensor = None,
                timestamps: torch.Tensor = None) -> torch.Tensor:
        B, T, F = observed.shape

        # Mask-aware input
        x = torch.cat([observed, mask], dim=-1)
        x = self.input_proj(x)
        x = self.input_norm(x)  # v3: stabilize
        x = self.dropout(x)

        # Apply encodings
        if self.use_temporal and timestamps is not None:
            x = self.temporal_enc(x, timestamps)
        if self.use_spatial and lat is not None and lon is not None:
            x = self.spatial_enc(x, lat, lon)
        if not self.use_spatial:
            x = self.pos_enc(x)

        # ── Parallel branches ──
        # Bi-LSTM
        lstm_out, _ = self.bilstm(x)
        lstm_feat = self.lstm_proj(lstm_out)            # (B, T, d_model)
        lstm_feat = self.lstm_norm(lstm_feat)            # v3: stabilize

        # Transformer
        trans_feat = self.transformer(x)                 # (B, T, d_model)

        # ── Gated fusion ──
        concat = torch.cat([lstm_feat, trans_feat], dim=-1)  # (B, T, 2*d_model)
        gate = self.gate(concat)                              # (B, T, d_model)
        fused = self.fusion(concat)                           # (B, T, d_model)
        x = gate * lstm_feat + (1 - gate) * trans_feat + fused

        # Output
        return self.output_proj(x)


class MABNetDeepTransformer(nn.Module):
    """MAB-Net with a deeper transformer (4-6 layers) but lighter Bi-LSTM.

    For scenarios where global patterns matter more (block missing).
    """

    def __init__(self, n_feats: int = 3, d_model: int = 128,
                 lstm_hidden: int = 64, n_head: int = 4,
                 n_transformer_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.n_feats = n_feats
        self.d_model = d_model

        # Mask-aware input
        self.input_proj = nn.Linear(2 * n_feats, d_model)
        self.pos_enc = SinusoidalPE(d_model)

        # Light Bi-LSTM (single layer)
        self.bilstm = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.lstm_proj = nn.Linear(lstm_hidden * 2, d_model)

        # Deep Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_transformer_layers)

        self.output_proj = nn.Linear(d_model, n_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, observed, mask, **kwargs):
        B, T, F = observed.shape
        x = torch.cat([observed, mask], dim=-1)
        x = self.input_proj(x)
        x = self.dropout(x)
        lstm_out, _ = self.bilstm(x)
        x = self.lstm_proj(lstm_out)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.output_proj(x)


# ══════════════════════════════════════════════════════════════════════════════
# Smoothness Loss
# ══════════════════════════════════════════════════════════════════════════════

class SmoothCompositeLoss(nn.Module):
    """L = MSE + α·MAE + β·Smoothness (R2's requested regularization).

    Smoothness penalizes large differences between adjacent time steps
    in the imputed values, encouraging physically realistic (smooth) output.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss(reduction="none")
        self.mae = nn.L1Loss(reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # Imputation loss (only on missing positions)
        mse_loss = (self.mse(pred, target) * mask).sum() / (mask.sum() + 1e-8)
        mae_loss = (self.mae(pred, target) * mask).sum() / (mask.sum() + 1e-8)
        imp_loss = mse_loss + self.alpha * mae_loss

        # Smoothness loss: penalize |pred[t] - pred[t-1]|
        diff = pred[:, 1:, :] - pred[:, :-1, :]  # (B, T-1, F)
        smooth_loss = diff.abs().mean()

        return imp_loss + self.beta * smooth_loss


# ══════════════════════════════════════════════════════════════════════════════
# Model Registry
# ══════════════════════════════════════════════════════════════════════════════

IMPROVED_MODELS = {
    "mab_net_parallel": MABNetParallel,
    "mab_net_deep":     MABNetDeepTransformer,
}


def build_improved_model(name: str, **kwargs) -> nn.Module:
    if name not in IMPROVED_MODELS:
        raise ValueError(f"Unknown: {name}. Options: {list(IMPROVED_MODELS)}")
    return IMPROVED_MODELS[name](**kwargs)