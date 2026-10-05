"""
Phase C Experiment Runner: Spatial & Temporal Encodings
========================================================
Based on Phase A+B best config: 168h window + mab_net_parallel.

Tests (all on block missing, 168h window):
  C1: +Spatial encoding (lat/lon)
  C2: +Temporal encoding (hour-of-day, day-of-year cyclic)
  C3: +Spatial + Temporal (both)

Plus spot-check on continuous missing to verify no regression.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))

from dataloader import (
    load_all_matrices, TRAIN_END, VAL_END, TEST_END, MASK_GENERATORS,
)
from models_v2 import build_improved_model, SmoothCompositeLoss

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(r"D:\tt\SCI\气象数据\processed_national_data")
SAVE_DIR = Path(r"D:\tt\SCI\experiments\results\phase_c")
DEVICE = torch.device("cpu")
N_STATIONS = 10
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
PATIENCE = 8
WINDOW_SIZE = 168        # ← best from Phase A
STRIDE = 168

PATTERNS = ["block", "continuous"]      # primary + spot-check
RATES = [0.30, 0.50, 0.80]

with open(DATA_DIR / "standardization_params.json") as f:
    STD_PARAMS = json.load(f)

# Load station coordinates for spatial encoding
import pandas as pd
coords_df = pd.read_csv(
    DATA_DIR / "station_coordinates.csv",
    names=["station_id", "latitude", "longitude"], header=0
)
STATION_LATS = torch.tensor(coords_df["latitude"].values[:N_STATIONS].astype(np.float32))
STATION_LONS = torch.tensor(coords_df["longitude"].values[:N_STATIONS].astype(np.float32))

print(f"Phase C: Spatial & Temporal Encodings")
print(f"Device: {DEVICE} | Window: {WINDOW_SIZE}h | Stations: {N_STATIONS}")
print(f"Station lat range: {STATION_LATS.min():.1f}-{STATION_LATS.max():.1f}")
print(f"Station lon range: {STATION_LONS.min():.1f}-{STATION_LONS.max():.1f}")


# ── Dataset with coordinates & timestamps ───────────────────────────────────

class SpatialTemporalDataset(Dataset):
    """Like _SplitDataset but returns lat/lon and timestamps for each station."""

    def __init__(self, data, mask, window_size, stride, station_lats, station_lons):
        self.data = torch.from_numpy(data)
        self.mask = mask
        self.window_size = window_size
        self.T, self.S, self.F = data.shape
        self.station_lats = torch.tensor(station_lats, dtype=torch.float32)
        self.station_lons = torch.tensor(station_lons, dtype=torch.float32)

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

        # timestamps: absolute hour index within 2024
        timestamps = torch.arange(t_start, t_end, dtype=torch.float32)

        return {
            "observed": observed,
            "mask": mask,
            "ground_truth": gt,
            "station_id": torch.tensor(s, dtype=torch.long),
            "t_start": torch.tensor(t_start, dtype=torch.long),
            "lat": self.station_lats[s],
            "lon": self.station_lons[s],
            "timestamps": timestamps,
        }


def build_st_loaders(data, mask_gen, rate, window_size, stride):
    """Build loaders that include spatial/temporal info."""
    T, S, F = data.shape
    train_mask = torch.from_numpy(mask_gen(TRAIN_END, F, rate, seed=42))
    val_mask   = torch.from_numpy(mask_gen(VAL_END - TRAIN_END, F, rate, seed=142))
    test_mask  = torch.from_numpy(mask_gen(TEST_END - VAL_END, F, rate, seed=242))

    sel = slice(0, N_STATIONS)
    lats = STATION_LATS[:N_STATIONS]
    lons = STATION_LONS[:N_STATIONS]

    train_ds = SpatialTemporalDataset(
        data[:TRAIN_END, sel, :], train_mask, window_size, stride, lats, lons
    )
    val_ds = SpatialTemporalDataset(
        data[TRAIN_END:VAL_END, sel, :], val_mask, window_size, stride, lats, lons
    )
    test_ds = SpatialTemporalDataset(
        data[VAL_END:TEST_END, sel, :], test_mask, window_size, stride=1,  # dense eval
        station_lats=lats, station_lons=lons
    )

    dl_kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=True, **dl_kw),
        DataLoader(val_ds, shuffle=False, **dl_kw),
        DataLoader(test_ds, shuffle=False, **dl_kw),
    )


def train_st_model(model, train_loader, val_loader, model_name, config, device, save_dir):
    """Training with spatial/temporal inputs."""
    epochs = config.get("epochs", 20)
    lr = config.get("lr", 1e-3)
    patience = config.get("patience", 8)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            obs = batch["observed"].to(device)
            msk = batch["mask"].to(device)
            gt  = batch["ground_truth"].to(device)
            lat = batch["lat"].to(device)
            lon = batch["lon"].to(device)
            ts  = batch["timestamps"].to(device)

            optimizer.zero_grad()
            pred = model(obs, msk, lat=lat, lon=lon, timestamps=ts)
            # Loss only on missing positions
            loss = (criterion(pred, gt) * msk).sum() / (msk.sum() + 1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch["observed"].to(device)
                msk = batch["mask"].to(device)
                gt  = batch["ground_truth"].to(device)
                lat = batch["lat"].to(device)
                lon = batch["lon"].to(device)
                ts  = batch["timestamps"].to(device)
                pred = model(obs, msk, lat=lat, lon=lon, timestamps=ts)
                batch_loss = ((criterion(pred, gt) * msk).sum() / (msk.sum() + 1e-8)).item()
                val_loss += batch_loss
                n_val += 1
        val_loss /= max(n_val, 1)
        scheduler.step(torch.tensor(val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    return {"best_val_loss": best_val_loss}


def eval_st_model(model, loader, device):
    """Evaluate model with spatial/temporal inputs."""
    model.eval()
    all_preds, all_gts, all_masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            obs = batch["observed"].to(device)
            msk = batch["mask"].to(device)
            gt  = batch["ground_truth"].to(device)
            lat = batch["lat"].to(device)
            lon = batch["lon"].to(device)
            ts  = batch["timestamps"].to(device)
            pred = model(obs, msk, lat=lat, lon=lon, timestamps=ts)
            all_preds.append(pred.cpu().numpy())
            all_gts.append(gt.cpu().numpy())
            all_masks.append(msk.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    gts   = np.concatenate(all_gts, axis=0)
    msks  = np.concatenate(all_masks, axis=0)
    return _compute_metrics(preds, gts, msks)


def _compute_metrics(predictions, ground_truth, masks):
    """MAE, RMSE, Pearson in physical units."""
    var_map = {0: "temperature", 1: "pressure", 2: "wind_speed"}
    var_label = ["TEM", "PRS", "WIN"]
    out = {}

    diff = predictions - ground_truth
    out["mae_std"] = float(np.abs(diff).mean())
    out["rmse_std"] = float(np.sqrt((diff**2).mean()))

    for f in range(3):
        mask_f = masks[:, :, f] == 1
        if mask_f.sum() < 2:
            continue
        v = var_map[f]
        pred_phys = predictions[:, :, f][mask_f] * STD_PARAMS[v]["std"] + STD_PARAMS[v]["mean"]
        gt_phys = ground_truth[:, :, f][mask_f] * STD_PARAMS[v]["std"] + STD_PARAMS[v]["mean"]
        diff_phys = pred_phys - gt_phys

        out[f"mae_{var_label[f]}"] = float(np.abs(diff_phys).mean())
        out[f"rmse_{var_label[f]}"] = float(np.sqrt((diff_phys**2).mean()))
        if len(pred_phys) > 1:
            out[f"pearson_{var_label[f]}"] = float(np.corrcoef(pred_phys, gt_phys)[0, 1])
        else:
            out[f"pearson_{var_label[f]}"] = 0.0
    return out


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()
    print("=" * 60)
    print("  Phase C: Spatial & Temporal Encodings")
    print("=" * 60)

    matrices, _, _ = load_all_matrices(DATA_DIR)
    data = np.stack([
        matrices["temperature_std"],
        matrices["pressure_std"],
        matrices["wind_speed_std"],
    ], axis=-1)

    # Model configs to test
    configs = [
        # (name, use_spatial, use_temporal, description)
        ("mab_net_parallel_baseline", False, False, "Baseline (no encoding)"),
        ("mab_net_parallel+spatial",    True,  False, "Spatial (lat/lon)"),
        ("mab_net_parallel+temporal",   False, True,  "Temporal (hour+doy)"),
        ("mab_net_parallel+spatial+temp", True, True, "Spatial + Temporal"),
    ]

    all_results = {"phase": "C", "window_size": WINDOW_SIZE, "experiments": []}

    for pattern in PATTERNS:
        mask_gen = MASK_GENERATORS[pattern]

        for rate in RATES:
            print(f"\n{'#'*55}")
            print(f"#  [{pattern} @ {rate:.0%}] — Window={WINDOW_SIZE}h")
            print(f"{'#'*55}")

            train_l, val_l, test_l = build_st_loaders(
                data, mask_gen, rate, WINDOW_SIZE, STRIDE
            )
            print(f"  Train samples: {len(train_l.dataset)}")

            for name, use_spatial, use_temporal, desc in configs:
                print(f"\n  ── {desc} ──")
                t1 = time.time()

                model = build_improved_model(
                    "mab_net_parallel",
                    use_spatial=use_spatial,
                    use_temporal=use_temporal,
                ).to(DEVICE)

                n_params = sum(p.numel() for p in model.parameters())
                print(f"  Model: {name} | Params: {n_params:,}")

                config = {"epochs": EPOCHS, "lr": LR, "patience": PATIENCE}
                subdir = SAVE_DIR / f"{pattern}_r{int(rate*100)}"
                subdir.mkdir(parents=True, exist_ok=True)

                train_st_model(model, train_l, val_l, name, config, DEVICE, subdir)

                # Evaluate
                test_m = eval_st_model(model, test_l, DEVICE)
                test_m["method"] = name
                test_m["pattern"] = pattern
                test_m["rate"] = rate
                test_m["use_spatial"] = use_spatial
                test_m["use_temporal"] = use_temporal
                test_m["window_size"] = WINDOW_SIZE
                test_m["time_s"] = round(time.time() - t1, 1)
                test_m["params"] = n_params
                all_results["experiments"].append(test_m)

                # Quick comparison
                mae_t = test_m.get("mae_TEM", -1)
                r_t = test_m.get("pearson_TEM", -1)
                print(f"  → MAE(TEM)={mae_t:.2f}°C | Pearson(TEM)={r_t:.3f} | {test_m['time_s']}s")

            # Save checkpoint
            with open(SAVE_DIR / "phase_c_partial.json", "w") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # ── Final ──
    total_min = (time.time() - t_total) / 60
    final_path = SAVE_DIR / "phase_c_results.json"
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*60}")
    print(f"  ✅ Phase C done! Total: {total_min:.0f} min")
    print(f"  Results: {final_path}")

    # Comparison table
    exps = all_results["experiments"]
    print(f"\n{'='*70}")
    print(f"  Phase C: Encoding Comparison (all {WINDOW_SIZE}h window, block missing)")
    print(f"  {'Encoding':<35} {'Rate':>4}  {'MAE(TEM)':>8} {'MAE(PRS)':>8} {'MAE(WIN)':>8} {'R(TEM)':>7}")
    print(f"  {'-'*69}")
    for r in sorted(exps, key=lambda x: (x["rate"], x.get("mae_TEM", 99))):
        use_s = "S" if r.get("use_spatial") else "-"
        use_t = "T" if r.get("use_temporal") else "-"
        label = f"  Spatial={use_s} Temporal={use_t}"
        print(f"  {label:<35} {r['rate']:>3.0%}  {r.get('mae_TEM',0):>8.2f} "
              f"{r.get('mae_PRS',0):>8.2f} {r.get('mae_WIN',0):>8.2f} "
              f"{r.get('pearson_TEM',0):>7.3f}")


if __name__ == "__main__":
    main()