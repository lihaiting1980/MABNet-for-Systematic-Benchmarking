"""
MAB-Net and Ablation Variants
==============================
- MABNet:        Full model (Bi-LSTM + Mask-Aware + Transformer)
- BiLSTMOnly:    Ablation — Bi-LSTM only, no Transformer
- TransformerOnly: Ablation — Transformer only, no Bi-LSTM
- MABNetNoMask:  Ablation — no mask-aware concatenation

Paper reference: MAB-Net uses composite loss = MSE + α·MAE (α=1).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Positional Encoding ─────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal PE from 'Attention Is All You Need'."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        return x + self.pe[:, :x.size(1), :]


# ── MAB-Net (Full Model) ────────────────────────────────────────────────────

class MABNet(nn.Module):
    """Mask-Aware Bidirectional Network for time-series imputation.

    Architecture (from paper §2.3):
    1. Mask-Aware Input Projection: concat(observed, mask) → Linear → d_model
    2. Bi-LSTM Encoder: captures local temporal dynamics
    3. Positional Encoding: sinusoidal
    4. Transformer Encoder: captures global dependencies
    5. Output Projection: Linear → n_feats
    """

    def __init__(self, n_feats: int = 3, d_model: int = 128,
                 lstm_hidden: int = 64, lstm_layers: int = 2,
                 n_head: int = 4, n_transformer_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.n_feats = n_feats
        self.d_model = d_model

        # 1. Mask-aware input projection: (2*n_feats) → d_model
        self.input_proj = nn.Linear(2 * n_feats, d_model)

        # 2. Bi-LSTM
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # Project LSTM output back to d_model for transformer
        self.lstm_proj = nn.Linear(lstm_out_dim, d_model)

        # 3. Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        # 4. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_transformer_layers)

        # 5. Output projection
        self.output_proj = nn.Linear(d_model, n_feats)

        self.dropout = nn.Dropout(dropout)

    def forward(self, observed: torch.Tensor, mask: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            observed: (B, T, F) — zero-filled input
            mask:     (B, T, F) — 1=missing, 0=observed
        Returns:
            imputed:  (B, T, F) — full imputed values
        """
        B, T, F = observed.shape

        # 1. Mask-aware input
        x = torch.cat([observed, mask], dim=-1)       # (B, T, 2F)
        x = self.input_proj(x)                         # (B, T, d_model)
        x = self.dropout(x)

        # 2. Bi-LSTM
        lstm_out, _ = self.bilstm(x)                   # (B, T, lstm_out_dim)
        x = self.lstm_proj(lstm_out)                   # (B, T, d_model)
        x = self.dropout(x)

        # 3. Positional encoding
        x = self.pos_enc(x)

        # 4. Transformer
        x = self.transformer(x)                        # (B, T, d_model)

        # 5. Output
        imputed = self.output_proj(x)                  # (B, T, F)

        return imputed


# ── Ablation 1: Bi-LSTM Only ────────────────────────────────────────────────

class BiLSTMOnly(nn.Module):
    """Bi-LSTM without mask-aware input or transformer."""

    def __init__(self, n_feats: int = 3, d_model: int = 128,
                 lstm_hidden: int = 64, lstm_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.n_feats = n_feats

        self.input_proj = nn.Linear(n_feats, d_model)  # no mask
        self.bilstm = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(lstm_hidden * 2, n_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(observed)                  # (B, T, d_model)
        x = self.dropout(x)
        lstm_out, _ = self.bilstm(x)
        return self.output_proj(lstm_out)


# ── Ablation 2: Transformer Only ────────────────────────────────────────────

class TransformerOnly(nn.Module):
    """Transformer-only, no Bi-LSTM."""

    def __init__(self, n_feats: int = 3, d_model: int = 128,
                 n_head: int = 4, n_transformer_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.n_feats = n_feats

        self.input_proj = nn.Linear(2 * n_feats, d_model)  # with mask
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head,
            dim_feedforward=d_model * 4, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_transformer_layers)
        self.output_proj = nn.Linear(d_model, n_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([observed, mask], dim=-1)
        x = self.input_proj(x)
        x = self.dropout(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.output_proj(x)


# ── Ablation 3: MAB-Net without Mask ────────────────────────────────────────

class MABNetNoMask(nn.Module):
    """Full MAB-Net architecture but WITHOUT mask concatenation (only observed)."""

    def __init__(self, n_feats: int = 3, d_model: int = 128,
                 lstm_hidden: int = 64, lstm_layers: int = 2,
                 n_head: int = 4, n_transformer_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.n_feats = n_feats
        self.d_model = d_model

        self.input_proj = nn.Linear(n_feats, d_model)   # only observed
        self.bilstm = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_proj = nn.Linear(lstm_hidden * 2, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head,
            dim_feedforward=d_model * 4, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_transformer_layers)
        self.output_proj = nn.Linear(d_model, n_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(observed)
        x = self.dropout(x)
        lstm_out, _ = self.bilstm(x)
        x = self.lstm_proj(lstm_out)
        x = self.dropout(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.output_proj(x)


# ── Model registry ──────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "mab_net":           MABNet,
    "bilstm_only":       BiLSTMOnly,
    "transformer_only":  TransformerOnly,
    "mab_net_no_mask":   MABNetNoMask,
}


def build_model(name: str, **kwargs) -> nn.Module:
    """Factory: build a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


# ── Composite Loss ──────────────────────────────────────────────────────────

class CompositeLoss(nn.Module):
    """L = MSE + α * MAE, computed only on missing positions."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction="none")
        self.mae = nn.L1Loss(reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Mask: 1=missing. Loss averaged over missing positions only."""
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        combined = mse_loss + self.alpha * mae_loss
        # Only count missing positions
        n_missing = mask.sum() + 1e-8
        return (combined * mask).sum() / n_missing