"""
MAB-Net Experiment — Data Loader
=================================
Loads the processed meteorological matrices, splits chronologically
(train Jan-Sep, val Oct, test Nov-Dec), generates missing masks
(random / continuous / block), and feeds sliding 24h windows.

Data source: 气象数据/processed_national_data/
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader


# ── Constants ──────────────────────────────────────────────────────────────
DATA_DIR = Path(r"D:\tt\SCI\气象数据\processed_national_data")
VARIABLES = ["temperature", "pressure", "wind_speed"]
WINDOW_SIZE = 24            # hours
STRIDE = 1                  # sliding window stride

# Chronological split (2024, leap year → 8784 hours)
# Jan:   744h  (rows   0 –  743)
# Feb:   672h  (rows 744 – 1415)
# Mar:   744h  (rows 1416 – 2159)
# Apr:   720h  (rows 2160 – 2879)
# May:   744h  (rows 2880 – 3623)
# Jun:   720h  (rows 3624 – 4343)
# Jul:   744h  (rows 4344 – 5087)
# Aug:   744h  (rows 5088 – 5831)
# Sep:   720h  (rows 5832 – 6551)   → train end
# Oct:   744h  (rows 6552 – 7295)   → val
# Nov:   720h  (rows 7296 – 8015)
# Dec:   744h  (rows 8016 – 8783)   → test
TRAIN_END = 6552   # exclusive
VAL_END   = 7296   # exclusive
TEST_END  = 8784   # exclusive


# ── Missing mask generators ─────────────────────────────────────────────────

def generate_random_mask(n_steps: int, n_feats: int, missing_rate: float,
                         seed: int = 42) -> np.ndarray:
    """Random missing: each (t, f) cell independently masked."""
    rng = np.random.RandomState(seed)
    return (rng.rand(n_steps, n_feats) < missing_rate).astype(np.float32)


def generate_continuous_mask(n_steps: int, n_feats: int, missing_rate: float,
                             seed: int = 42) -> np.ndarray:
    """Continuous missing: randomly placed gaps of ~6h mean length.

    We repeatedly pick a random start position and mask `gap_len` consecutive
    hours, until the desired overall missing rate is reached.
    """
    rng = np.random.RandomState(seed)
    mask = np.zeros((n_steps, n_feats), dtype=np.float32)
    target_missing = int(n_steps * n_feats * missing_rate)
    current = 0
    mean_gap = 6
    while current < target_missing:
        gap_len = rng.poisson(mean_gap)
        if gap_len < 1:
            gap_len = 1
        start = rng.randint(0, max(1, n_steps - gap_len))
        feat = rng.randint(0, n_feats)
        end = min(start + gap_len, n_steps)
        for t in range(start, end):
            if mask[t, feat] == 0:
                mask[t, feat] = 1
                current += 1
                if current >= target_missing:
                    break
    return mask


def generate_block_mask(n_steps: int, n_feats: int, missing_rate: float,
                        seed: int = 42) -> np.ndarray:
    """Block missing: long consecutive gaps (days) across ALL features.

    We place a small number of long blocks.  For 80% rate on 8784 hours that
    means e.g. 2-3 blocks of ~2300-3500 hours each.
    """
    rng = np.random.RandomState(seed)
    mask = np.zeros((n_steps, n_feats), dtype=np.float32)
    target_missing = int(n_steps * n_feats * missing_rate)
    current = 0
    while current < target_missing:
        # block length ~ proportion of remaining time
        remaining_ratio = (target_missing - current) / (n_steps * n_feats)
        block_len = int(n_steps * remaining_ratio * 0.45)  # avoid too few blocks
        block_len = max(24, min(block_len, n_steps // 2))
        start = rng.randint(0, max(1, n_steps - block_len))
        for t in range(start, min(start + block_len, n_steps)):
            for f in range(n_feats):
                if mask[t, f] == 0:
                    mask[t, f] = 1
                    current += 1
                    if current >= target_missing:
                        return mask
    return mask


MASK_GENERATORS = {
    "random":     generate_random_mask,
    "continuous": generate_continuous_mask,
    "block":      generate_block_mask,
}


# ── Data loading ────────────────────────────────────────────────────────────

def load_all_matrices(data_dir: Path = DATA_DIR) -> Dict[str, np.ndarray]:
    """Load filled + standardized matrices for 3 variables.

    Returns dict with keys like 'temperature_filled', 'temperature_std', etc.
    Also returns standardization params and station info.
    """
    matrices = {}

    for var in VARIABLES:
        for suffix, key in [("filled_matrix", f"{var}_filled"),
                            ("standardized_matrix", f"{var}_std")]:
            fpath = data_dir / f"{var}_{suffix}.csv"
            df = pd.read_csv(fpath, index_col=0)
            matrices[key] = df.values.astype(np.float32)  # shape (8784, 405)

    # station info
    station_info = pd.read_csv(data_dir / "station_info.csv")
    station_coords = pd.read_csv(
        data_dir / "station_coordinates.csv",
        names=["station_id", "latitude", "longitude"], header=0
    )

    return matrices, station_info, station_coords


def build_samples(matrices: Dict[str, np.ndarray],
                  split: str = "train",
                  window_size: int = WINDOW_SIZE,
                  stride: int = STRIDE) -> np.ndarray:
    """Extract sliding windows for one data split.

    Returns array of shape (n_samples, window_size, n_feats, n_stations).
    The paper treats each station independently, so we reshape below.

    Actually, for compatibility with the PyTorch Dataset that samples
    station-window pairs, we keep (T, stations, features) and the Dataset
    will index into it.
    """
    if split == "train":
        t_start, t_end = 0, TRAIN_END
    elif split == "val":
        t_start, t_end = TRAIN_END, VAL_END
    elif split == "test":
        t_start, t_end = VAL_END, TEST_END
    else:  # "all"
        t_start, t_end = 0, TEST_END

    # Stack standardized matrices: (T, stations, features)
    # Order: TEM, PRS, WIN
    data = np.stack([
        matrices["temperature_std"][t_start:t_end],
        matrices["pressure_std"][t_start:t_end],
        matrices["wind_speed_std"][t_start:t_end],
    ], axis=-1)  # (T_split, 405, 3)

    return data


class ImputationDataset(Dataset):
    """PyTorch Dataset for time-series imputation.

    For each station independently, we take sliding windows of `window_size`
    hours.  The model imputes the masked positions.

    Parameters
    ----------
    data : np.ndarray of shape (T, n_stations, n_feats)
    mask_pattern : "random" | "continuous" | "block"
    missing_rate : float in [0.0, 1.0]
    window_size : int
    stride : int — sampling stride (use >1 for efficiency)
    mask_seed : int — determinism for mask generation
    """

    def __init__(self, data: np.ndarray, mask_pattern: str = "random",
                 missing_rate: float = 0.5, window_size: int = WINDOW_SIZE,
                 stride: int = STRIDE, mask_seed: int = 42):
        super().__init__()
        self.data = torch.from_numpy(data)  # (T, S, F)
        self.T, self.S, self.F = data.shape
        self.window_size = window_size
        self.stride = stride
        self.mask_pattern = mask_pattern
        self.missing_rate = missing_rate
        self.mask_seed = mask_seed

        # Pre-generate a global mask for the full time range, then slice
        mask_gen = MASK_GENERATORS[mask_pattern]
        # station-agnostic mask: apply same temporal pattern across stations
        self.global_mask = torch.from_numpy(
            mask_gen(self.T, self.F, missing_rate, mask_seed)
        )  # (T, F)

        # Build index: all (station, window_start) pairs
        self.indices = []
        for s in range(self.S):
            for t in range(0, self.T - window_size + 1, stride):
                self.indices.append((s, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s, t_start = self.indices[idx]
        t_end = t_start + self.window_size

        # Ground truth (window_size, F)
        gt = self.data[t_start:t_end, s, :]

        # Mask for this window
        mask = self.global_mask[t_start:t_end, :]  # (window_size, F)

        # Observed = gt where mask==0, else 0
        observed = gt.clone()
        observed[mask.bool()] = 0.0

        return {
            "observed": observed,       # (W, F)
            "mask": mask,               # (W, F)  — 1 = missing
            "ground_truth": gt,         # (W, F)
            "station_id": s,
            "t_start": t_start,
        }


def get_dataloaders(data: np.ndarray, mask_pattern: str, missing_rate: float,
                    batch_size: int = 256, num_workers: int = 0,
                    window_size: int = WINDOW_SIZE,
                    stride: int = STRIDE) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders for one experiment config."""
    # We share a global mask across all splits.
    # The data already split chronologically — the mask is applied per-split.
    T_total, S, F = data.shape
    train_mask_gen = MASK_GENERATORS[mask_pattern]
    val_mask_gen   = MASK_GENERATORS[mask_pattern]
    test_mask_gen  = MASK_GENERATORS[mask_pattern]

    # We need per-split masks so that the mask patterns are independent
    # across splits (otherwise the model "sees" the test mask pattern during
    # training via the mask structure).  Use different seeds.
    train_mask = torch.from_numpy(train_mask_gen(TRAIN_END, F, missing_rate, 42))
    val_mask   = torch.from_numpy(val_mask_gen(VAL_END - TRAIN_END, F, missing_rate, 142))
    test_mask  = torch.from_numpy(test_mask_gen(TEST_END - VAL_END, F, missing_rate, 242))

    train_ds = _SplitDataset(data[:TRAIN_END], train_mask, window_size, stride)
    val_ds   = _SplitDataset(data[TRAIN_END:VAL_END], val_mask, window_size, stride)
    test_ds  = _SplitDataset(data[VAL_END:TEST_END], test_mask, window_size, stride)

    dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                     pin_memory=torch.cuda.is_available(), shuffle=True)
    return (DataLoader(train_ds, **dl_kwargs),
            DataLoader(val_ds, **dl_kwargs),
            DataLoader(test_ds, **dl_kwargs))


class _SplitDataset(Dataset):
    """Internal: one chronological split with a pre-generated mask."""
    def __init__(self, data: np.ndarray, mask: torch.Tensor,
                 window_size: int, stride: int):
        self.data = torch.from_numpy(data)  # (T, S, F)
        self.mask = mask                     # (T, F) — 1 = missing
        self.window_size = window_size
        self.T, self.S, self.F = data.shape
        self.indices = []
        for s in range(self.S):
            for t in range(0, self.T - window_size + 1, stride):
                self.indices.append((s, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s, t_start = self.indices[idx]
        t_end = t_start + self.window_size
        gt = self.data[t_start:t_end, s, :]
        mask = self.mask[t_start:t_end, :]
        observed = gt.clone()
        observed[mask.bool()] = 0.0
        return {
            "observed": observed,
            "mask": mask,
            "ground_truth": gt,
            "station_id": s,
            "t_start": t_start,
        }