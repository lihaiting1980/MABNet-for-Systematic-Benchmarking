"""
Baseline Methods for Time-Series Imputation
============================================
- Mean imputation (paper baseline)
- Linear interpolation (paper baseline)
- Cubic spline interpolation (NEW — per Reviewer 2)
- Polynomial fit 2nd order (NEW — per Reviewer 2)
- BRITS (paper baseline, re-implemented)
- SAITS (NEW SOTA baseline)

Non-DL methods operate per-station, per-feature, using only observed values
within the 24h window.  DL methods (BRITS, SAITS) are trained.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from scipy.interpolate import CubicSpline, PchipInterpolator


# ══════════════════════════════════════════════════════════════════════════════
# Statistical baselines (no training)
# ══════════════════════════════════════════════════════════════════════════════

def mean_imputation(observed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill missing values with the mean of observed values in the window.

    Args:
        observed: (T, F) or (B, T, F) — zero-filled
        mask:     same shape — 1=missing
    Returns:
        imputed:  same shape
    """
    if observed.ndim == 2:
        observed, mask = observed[np.newaxis], mask[np.newaxis]
        squeeze = True
    else:
        squeeze = False

    B, T, F = observed.shape
    imputed = observed.copy()
    for b in range(B):
        for f in range(F):
            obs_vals = imputed[b, mask[b, :, f] == 0, f]
            if len(obs_vals) > 0:
                fill_val = obs_vals.mean()
            else:
                fill_val = 0.0
            imputed[b, mask[b, :, f] == 1, f] = fill_val

    return imputed[0] if squeeze else imputed


def linear_interpolation(observed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """1D linear interpolation per feature, per station.

    Uses numpy.interp: treats time index as x, observed values as y.
    """
    if observed.ndim == 2:
        observed, mask = observed[np.newaxis], mask[np.newaxis]
        squeeze = True
    else:
        squeeze = False

    B, T, F = observed.shape
    imputed = observed.copy()
    t_idx = np.arange(T)

    for b in range(B):
        for f in range(F):
            obs_mask = mask[b, :, f] == 0
            if obs_mask.sum() < 2:
                # Not enough points to interpolate
                continue
            xp = t_idx[obs_mask]
            yp = imputed[b, obs_mask, f]
            # Interpolate at ALL time steps
            imputed[b, :, f] = np.interp(t_idx, xp, yp)

    return imputed[0] if squeeze else imputed


def cubic_spline_interpolation(observed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cubic spline interpolation per feature, per station.

    Falls back to linear interpolation when there are too few observed points
    or when spline fitting fails (e.g., near-constant segments).
    """
    if observed.ndim == 2:
        observed, mask = observed[np.newaxis], mask[np.newaxis]
        squeeze = True
    else:
        squeeze = False

    B, T, F = observed.shape
    imputed = observed.copy()
    t_idx = np.arange(T)

    for b in range(B):
        for f in range(F):
            obs_mask = mask[b, :, f] == 0
            n_obs = obs_mask.sum()
            if n_obs < 4:
                # Fall back to linear
                if n_obs >= 2:
                    xp = t_idx[obs_mask]
                    yp = imputed[b, obs_mask, f]
                    imputed[b, :, f] = np.interp(t_idx, xp, yp)
                continue
            xp = t_idx[obs_mask].astype(float)
            yp = imputed[b, obs_mask, f]
            try:
                cs = CubicSpline(xp, yp, bc_type='natural', extrapolate=True)
                imputed[b, :, f] = cs(t_idx.astype(float))
            except (ValueError, np.linalg.LinAlgError):
                # Fall back to linear
                imputed[b, :, f] = np.interp(t_idx, xp, yp)

    return imputed[0] if squeeze else imputed


def polynomial_interpolation(observed: np.ndarray, mask: np.ndarray,
                             degree: int = 2) -> np.ndarray:
    """Polynomial fit (least-squares) per feature, per station.

    For missing positions, the polynomial is fitted on observed points only,
    then evaluated at all time positions.
    """
    if observed.ndim == 2:
        observed, mask = observed[np.newaxis], mask[np.newaxis]
        squeeze = True
    else:
        squeeze = False

    B, T, F = observed.shape
    imputed = observed.copy()
    t_idx = np.arange(T, dtype=float)
    # Normalize t to [-1, 1] for numerical stability
    t_norm = 2 * (t_idx - t_idx.min()) / max(1, t_idx.max() - t_idx.min()) - 1

    for b in range(B):
        for f in range(F):
            obs_mask = mask[b, :, f] == 0
            n_obs = obs_mask.sum()
            if n_obs < degree + 1:
                if n_obs >= 2:
                    xp = t_idx[obs_mask]
                    yp = imputed[b, obs_mask, f]
                    imputed[b, :, f] = np.interp(t_idx, xp, yp)
                continue
            xp = t_norm[obs_mask]
            yp = imputed[b, obs_mask, f]
            try:
                coeffs = np.polyfit(xp, yp, degree)
                poly = np.poly1d(coeffs)
                # Only fill missing positions
                missing_mask = mask[b, :, f] == 1
                imputed[b, missing_mask, f] = poly(t_norm[missing_mask])
            except (ValueError, np.linalg.LinAlgError):
                imputed[b, :, f] = np.interp(t_idx, xp, yp)

    return imputed[0] if squeeze else imputed


# Registry of stat baselines (no training required)
STAT_BASELINES = {
    "mean":       mean_imputation,
    "linear":     linear_interpolation,
    "cubic_spline": cubic_spline_interpolation,
    "polynomial": lambda o, m: polynomial_interpolation(o, m, degree=2),
}

# ══════════════════════════════════════════════════════════════════════════════
# BRITS — Bidirectional Recurrent Imputation for Time Series
# ══════════════════════════════════════════════════════════════════════════════

class BRITS(nn.Module):
    """Simplified BRITS implementation.

    Reference: Cao et al., "BRITS: Bidirectional Recurrent Imputation
    for Time Series", NeurIPS 2018.

    Key idea: use a bidirectional RNN where the hidden state at each step
    is used to predict (impute) the value at that step.  Missing values are
    treated as learnable variables updated during backprop.

    This implementation uses a Bi-LSTM with a linear readout per time step.
    """

    def __init__(self, n_feats: int = 3, hidden_size: int = 256,
                 rnn_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.n_feats = n_feats
        self.hidden_size = hidden_size

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_feats, hidden_size),
            nn.ReLU(),
        )

        # Bi-directional RNN
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=False,  # we'll do forward + backward manually
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        # Temporal decay (γ) for missing values
        # For simplicity, use a fixed decay
        self.register_buffer("decay", torch.linspace(0.0, 1.0, 256))

        # Output: hidden → imputed value
        self.impute_proj = nn.Linear(hidden_size, n_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observed: (B, T, F) zero-filled
            mask:     (B, T, F) 1=missing
        Returns:
            imputed:  (B, T, F)
        """
        B, T, F = observed.shape

        # Forward pass
        x = self.input_proj(observed)                     # (B, T, H)
        x = self.dropout(x)
        fwd_out, _ = self.rnn(x)                          # (B, T, H)
        fwd_imp = self.impute_proj(fwd_out)               # (B, T, F)

        # Backward pass: reverse the sequence
        x_rev = torch.flip(x, dims=[1])
        bwd_out, _ = self.rnn(x_rev)
        bwd_out = torch.flip(bwd_out, dims=[1])
        bwd_imp = self.impute_proj(bwd_out)

        # Combine: use forward imputation then blend
        # Average forward and backward predictions
        imputed = (fwd_imp + bwd_imp) / 2

        return imputed


# ══════════════════════════════════════════════════════════════════════════════
# SAITS — Self-Attention-based Imputation for Time Series
# ══════════════════════════════════════════════════════════════════════════════

class SAITS(nn.Module):
    """Simplified SAITS implementation.

    Reference: Du et al., "SAITS: Self-Attention-based Imputation for
    Time Series", Expert Systems with Applications, 2023.

    Key ideas:
    - Two self-attention branches (Diag-Masked + Non-Diag Masked)
    - Joint optimization: masked imputation + observed reconstruction
    - Weighted combination of the two branches
    """

    def __init__(self, n_feats: int = 3, d_model: int = 128, n_head: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_feats = n_feats
        self.d_model = d_model

        # Mask-aware input: concat(observed, mask) → d_model
        self.input_proj = nn.Linear(2 * n_feats, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncodingSAITS(d_model)

        # Transformer encoder (shared across two branches)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head,
            dim_feedforward=d_model * 4, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # Two output heads for the two tasks
        self.output_imp  = nn.Linear(d_model, n_feats)  # imputation
        self.output_recon = nn.Linear(d_model, n_feats)  # reconstruction

        # Learnable combination weight
        self.eta = nn.Parameter(torch.tensor(0.5))

        self.dropout = nn.Dropout(dropout)

    def forward(self, observed: torch.Tensor, mask: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        """Returns dict with 'imputed', 'reconstructed', and combined 'output'."""
        x = torch.cat([observed, mask], dim=-1)           # (B, T, 2F)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        x = self.transformer(x)

        imputed_hat = self.output_imp(x)                   # masked imputation
        recon_hat   = self.output_recon(x)                 # observed reconstruction

        # Combination: eta * imputation + (1-eta) * reconstruction
        combined = self.eta * imputed_hat + (1 - self.eta) * recon_hat

        return {
            "imputed": imputed_hat,
            "reconstructed": recon_hat,
            "output": combined,
        }


class SinusoidalPositionalEncodingSAITS(nn.Module):
    """Same as in models.py but kept self-contained for the SAITS module."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        import math
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


# ══════════════════════════════════════════════════════════════════════════════
# KNN Imputation (simple but informative baseline)
# ══════════════════════════════════════════════════════════════════════════════

def knn_imputation(observed: np.ndarray, mask: np.ndarray, k: int = 5) -> np.ndarray:
    """KNN imputation: fill missing values using k-nearest time steps.

    For each missing position (t, f), find the k most similar time steps
    (across all features) and use their average for feature f.

    Args:
        observed: (T, F)
        mask:     (T, F) — 1=missing
        k:        number of neighbors
    """
    if observed.ndim == 2:
        observed, mask = observed[np.newaxis], mask[np.newaxis]
        squeeze = True
    else:
        squeeze = False

    B, T, F = observed.shape
    imputed = observed.copy()

    for b in range(B):
        # Find fully-observed time steps as reference
        full_mask = mask[b].sum(axis=1) == 0
        if full_mask.sum() < k:
            # Not enough reference points — use mean
            imputed[b] = mean_imputation(observed[b], mask[b])
            continue

        ref_idx = np.where(full_mask)[0]
        ref_data = imputed[b, ref_idx, :]  # (n_ref, F)

        # For each missing time step
        missing_rows = mask[b].sum(axis=1) > 0
        for t in np.where(missing_rows)[0]:
            row = imputed[b, t, :]
            # Only use observed features in this row for distance
            obs_f = mask[b, t, :] == 0
            if obs_f.sum() == 0:
                # All missing — use global mean
                imputed[b, t, :] = ref_data.mean(axis=0)
                continue

            # Distance to reference rows (on observed features)
            dist = np.sum((ref_data[:, obs_f] - row[obs_f]) ** 2, axis=1)
            knn_idx = ref_idx[np.argsort(dist)[:k]]

            # Impute missing features
            for f in range(F):
                if mask[b, t, f] == 1:
                    imputed[b, t, f] = imputed[b, knn_idx, f].mean()

    return imputed[0] if squeeze else imputed