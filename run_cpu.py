"""
CPU-Friendly Experiment Runner
===============================
Reduced-scale training suitable for CPU. Uses:
  - 20 stations (out of 405)
  - stride=24 (1 sample per day, 24× reduction)
  - batch_size=32
  - 20 epochs

Covers: 3 missing patterns × 3 rates × 7 methods (MAB-Net + 3 ablation + BRITS + SAITS)
Plus: statistical baselines for comparison.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from dataloader import (
    load_all_matrices, _SplitDataset,
    TRAIN_END, VAL_END, TEST_END, WINDOW_SIZE, MASK_GENERATORS,
)
from models import build_model, CompositeLoss
from baselines import STAT_BASELINES, BRITS, SAITS
from train import train_model, evaluate, CompositeLossDL

# ── CPU-friendly config ─────────────────────────────────────────────────────
DATA_DIR = Path(r"D:\tt\SCI\气象数据\processed_national_data")
SAVE_DIR = Path(r"D:\tt\SCI\experiments\results")
DEVICE = torch.device("cpu")
N_STATIONS = 10        # ← reduced from 405
BATCH_SIZE = 32        # ← smaller batch
EPOCHS = 15            # ← sufficient for convergence
LR = 1e-3
PATIENCE = 6
STRIDE = 24            # ← 1 sample per day instead of every hour
MISSING_PATTERNS = ["random", "continuous", "block"]
MISSING_RATES = [0.30, 0.50, 0.80]

with open(DATA_DIR / "standardization_params.json") as f:
    STD_PARAMS = json.load(f)

print(f"Device: {DEVICE} | Stations: {N_STATIONS} | Stride: {STRIDE}h | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")


# ── Helpers ─────────────────────────────────────────────────────────────────

def denormalize(data, var):
    return data * STD_PARAMS[var]["std"] + STD_PARAMS[var]["mean"]


def compute_metrics(predictions, ground_truth, masks):
    """MAE, RMSE, Pearson in physical units per variable."""
    var_map = {0: "temperature", 1: "pressure", 2: "wind_speed"}
    var_label = ["TEM", "PRS", "WIN"]
    out = {}

    diff = predictions - ground_truth
    out["mae_std"] = float(np.abs(diff).mean())
    out["rmse_std"] = float(np.sqrt((diff ** 2).mean()))

    for f in range(3):
        mask_f = masks[:, :, f] == 1
        if mask_f.sum() < 2:
            continue
        pred_phys = denormalize(predictions[:, :, f][mask_f], var_map[f])
        gt_phys   = denormalize(ground_truth[:, :, f][mask_f], var_map[f])
        diff_phys = pred_phys - gt_phys

        out[f"mae_{var_label[f]}"]  = float(np.abs(diff_phys).mean())
        out[f"rmse_{var_label[f]}"] = float(np.sqrt((diff_phys ** 2).mean()))
        out[f"pearson_{var_label[f]}"] = float(
            np.corrcoef(pred_phys, gt_phys)[0, 1] if len(pred_phys) > 1 else 0
        )
    return out


def build_dl_loaders(data, mask_gen, rate, shuffle_train=True):
    """Build train/val/test DataLoaders for one config."""
    T, S, F = data.shape
    train_mask = torch.from_numpy(mask_gen(TRAIN_END, F, rate, seed=42))
    val_mask   = torch.from_numpy(mask_gen(VAL_END - TRAIN_END, F, rate, seed=142))
    test_mask  = torch.from_numpy(mask_gen(TEST_END - VAL_END, F, rate, seed=242))

    # Subsample stations
    sel_stations = slice(0, N_STATIONS)
    data_train = data[:TRAIN_END, sel_stations, :]
    data_val   = data[TRAIN_END:VAL_END, sel_stations, :]
    data_test  = data[VAL_END:TEST_END, sel_stations, :]

    train_ds = _SplitDataset(data_train, train_mask, WINDOW_SIZE, STRIDE)
    val_ds   = _SplitDataset(data_val, val_mask, WINDOW_SIZE, STRIDE)
    test_ds  = _SplitDataset(data_test, test_mask, WINDOW_SIZE, stride=1)  # dense eval

    dl_kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=shuffle_train, **dl_kw),
        DataLoader(val_ds, shuffle=False, **dl_kw),
        DataLoader(test_ds, shuffle=False, **dl_kw),
    )


def eval_dl(model, loader, is_saits=False):
    """Evaluate trained DL model."""
    model.eval()
    all_preds, all_gts, all_masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            obs = batch["observed"].to(DEVICE)
            msk = batch["mask"].to(DEVICE)
            gt  = batch["ground_truth"].to(DEVICE)
            pred = model(obs, msk)
            if is_saits:
                pred = pred["output"]
            all_preds.append(pred.cpu().numpy())
            all_gts.append(gt.cpu().numpy())
            all_masks.append(msk.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    gts   = np.concatenate(all_gts, axis=0)
    msks  = np.concatenate(all_masks, axis=0)
    return compute_metrics(preds, gts, msks)


def eval_stat(data_test, mask_gen, rate):
    """Evaluate one statistical baseline."""
    T, S, F = data_test.shape
    test_mask = mask_gen(T, F, rate, seed=242)
    results = {}
    for name, func in STAT_BASELINES.items():
        all_preds, all_gts, all_masks = [], [], []
        for s in range(min(S, N_STATIONS)):
            gt = data_test[:, s, :]
            obs = gt.copy()
            obs[test_mask.astype(bool)] = 0.0
            try:
                pred = func(obs, test_mask)
                all_preds.append(pred[np.newaxis])
                all_gts.append(gt[np.newaxis])
                all_masks.append(test_mask[np.newaxis])
            except Exception as e:
                continue
        if all_preds:
            p = np.concatenate(all_preds, axis=0)
            g = np.concatenate(all_gts, axis=0)
            m = np.concatenate(all_masks, axis=0)
            results[name] = compute_metrics(p, g, m)
        else:
            results[name] = {"error": "all stations failed"}
    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CPU-Friendly Experiment Runner")
    print("=" * 60)

    # Load data
    t0 = time.time()
    print("\n[1/3] Loading data...")
    matrices, _, _ = load_all_matrices(DATA_DIR)
    data = np.stack([
        matrices["temperature_std"],
        matrices["pressure_std"],
        matrices["wind_speed_std"],
    ], axis=-1)
    print(f"  Data: {data.shape} | Load time: {time.time()-t0:.1f}s")

    all_results = {"config": {"n_stations": N_STATIONS, "stride": STRIDE,
                              "batch_size": BATCH_SIZE, "epochs": EPOCHS},
                   "stat_baselines": [],
                   "dl_models": [],
                   "ablation": []}
    run_id = 0
    total_runs = len(MISSING_PATTERNS) * len(MISSING_RATES) * 7  # 5 stat + 7 DL

    # DL model names
    dl_names = ["brits", "saits", "mab_net", "bilstm_only",
                "transformer_only", "mab_net_no_mask"]

    # Pre-load test split for stat baselines
    data_test = data[VAL_END - VAL_END:TEST_END - VAL_END]

    for pattern in MISSING_PATTERNS:
        mask_gen = MASK_GENERATORS[pattern]
        for rate in MISSING_RATES:
            print(f"\n{'─'*55}")
            print(f"  [{pattern} @ {rate:.0%}]")
            print(f"{'─'*55}")

            # ── Statistical baselines ──
            print("  Statistical baselines...", end=" ", flush=True)
            stat_r = eval_stat(data_test, mask_gen, rate)
            for name, m in stat_r.items():
                m["method"] = name
                m["pattern"] = pattern
                m["rate"] = rate
                all_results["stat_baselines"].append(m)
            best_stat = min(v.get("mae_std", 999) for v in stat_r.values())
            print(f"best MAE(std) = {best_stat:.4f}")

            # ── Build loaders ──
            train_loader, val_loader, test_loader = build_dl_loaders(
                data, mask_gen, rate
            )
            n_train = len(train_loader.dataset)
            n_test  = len(test_loader.dataset)
            print(f"  Samples: train={n_train}, test={n_test}")

            # ── Train DL models ──
            for name in dl_names:
                run_id += 1
                is_saits = (name == "saits")
                print(f"\n  [{run_id}/{total_runs}] Training {name}...", flush=True)

                t1 = time.time()
                # Build model
                if name == "brits":
                    model = BRITS().to(DEVICE)
                elif name == "saits":
                    model = SAITS().to(DEVICE)
                else:
                    model = build_model(name).to(DEVICE)

                config = {"epochs": EPOCHS, "lr": LR, "patience": PATIENCE}
                subdir = SAVE_DIR / "cpu_runs" / f"{pattern}_rate{int(rate*100)}"
                subdir.mkdir(parents=True, exist_ok=True)

                train_result = train_model(model, train_loader, val_loader,
                                          name, config, DEVICE, subdir)
                test_metrics = eval_dl(model, test_loader, is_saits)
                test_metrics["method"] = name
                test_metrics["pattern"] = pattern
                test_metrics["rate"] = rate
                test_metrics["train_time_s"] = round(time.time() - t1, 1)
                test_metrics["best_epoch"] = train_result["best_epoch"]

                # Categorize
                if name in ("brits", "saits"):
                    all_results["dl_models"].append(test_metrics)
                else:
                    all_results["ablation"].append(test_metrics)

                elapsed = time.time() - t1
                eta = elapsed * (total_runs - run_id) / 60
                print(f"    Done in {elapsed:.0f}s | "
                      f"MAE(TEM)={test_metrics.get('mae_TEM', -1):.3f}°C | "
                      f"EST remaining: {eta:.0f}min")

            # ── Save checkpoint after each config ──
            ckpt_path = SAVE_DIR / "cpu_results_partial.json"
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # ── Final save ──
    final_path = SAVE_DIR / "cpu_results.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    total_time = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"  ✅ All done! Total time: {total_time:.0f} min")
    print(f"  Results: {final_path}")
    print(f"{'='*60}")

    # Print summary
    print_summary(all_results)


def print_summary(results):
    """Compact results table."""
    print("\n" + "=" * 85)
    print("  RESULTS TABLE (MAE in physical units: °C / hPa / m/s)")
    print("=" * 85)
    hdr = f"{'Method':<18} {'Pattern':<12} {'Rate':>4}  {'TEM':>8} {'PRS':>8} {'WIN':>8} {'R(T)':>7} {'Epoch':>5} {'Time':>7}"
    print(hdr)
    print("-" * 85)

    for cat, entries in [("Baseline", results.get("stat_baselines", [])),
                          ("DL", results.get("dl_models", [])),
                          ("Ablation", results.get("ablation", []))]:
        if entries:
            print(f"  [{cat}]")
        for r in entries:
            if "error" in r:
                print(f"  {r['method']:<16} ERROR: {r['error']}")
                continue
            print(f"  {r.get('method','?'):<16} {r.get('pattern',''):<12} "
                  f"{r.get('rate',0):>3.0%}  "
                  f"{r.get('mae_TEM',0):>8.3f} {r.get('mae_PRS',0):>8.3f} "
                  f"{r.get('mae_WIN',0):>8.3f} {r.get('pearson_TEM',0):>7.3f} "
                  f"{r.get('best_epoch','-'):>5} {r.get('train_time_s','-'):>6}")


if __name__ == "__main__":
    main()