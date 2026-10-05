"""
MAB-Net Experiment Runner
==========================
Orchestrates ALL experiments:
  (A) Baseline comparison: 6 methods × 3 patterns × 3 rates = 54 configs
  (B) Ablation study: 4 model variants × 3 patterns × 3 rates = 36 configs

Also computes physical-unit metrics by inverse-normalizing predictions.

Usage:
    python runner.py                    # run all experiments
    python runner.py --baseline-only    # baselines only
    python runner.py --ablation-only    # ablation only
    python runner.py --quick            # single config for testing
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Project modules
from dataloader import (
    load_all_matrices, build_samples, _SplitDataset,
    TRAIN_END, VAL_END, TEST_END, WINDOW_SIZE, MASK_GENERATORS,
)
from models import build_model, CompositeLoss
from baselines import (
    STAT_BASELINES, BRITS, SAITS, knn_imputation,
)
from train import train_model, evaluate, CompositeLossDL


# ── Config ──────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(r"D:\tt\SCI\气象数据\processed_national_data")
SAVE_DIR = Path(r"D:\tt\SCI\experiments\results")
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
PATIENCE = 10
STRIDE = 4          # stride for DL training (reduce #samples)

MISSING_PATTERNS = ["random", "continuous", "block"]
MISSING_RATES = [0.30, 0.50, 0.80]

# Standardization params (for physical-unit metrics)
with open(DATA_DIR / "standardization_params.json") as f:
    STD_PARAMS = json.load(f)

VAR_NAMES = ["TEM", "PRS", "WIN"]

print(f"Device: {DEVICE}")
print(f"Results dir: {SAVE_DIR.absolute()}")


# ── Helpers ─────────────────────────────────────────────────────────────────

def denormalize(data: np.ndarray, variable: str) -> np.ndarray:
    """Convert standardized values back to physical units."""
    mean = STD_PARAMS[variable]["mean"]
    std  = STD_PARAMS[variable]["std"]
    return data * std + mean


def compute_physical_metrics(predictions: np.ndarray, ground_truth: np.ndarray,
                             masks: np.ndarray) -> dict:
    """Compute MAE, RMSE, R² in physical units per variable.

    Args:
        predictions: (N, T, F) — imputed values (standardized)
        ground_truth: (N, T, F) — true values (standardized)
        masks: (N, T, F) — 1=missing
    """
    var_map = {0: "temperature", 1: "pressure", 2: "wind_speed"}
    metrics = {}

    # Overall standardized metrics
    diff = predictions - ground_truth
    mae_std = np.abs(diff).mean()
    rmse_std = np.sqrt((diff ** 2).mean())
    metrics["mae_std"] = float(mae_std)
    metrics["rmse_std"] = float(rmse_std)

    # Per-variable physical-unit metrics
    for f in range(3):
        var = var_map[f]
        feat_pred = predictions[:, :, f]
        feat_gt   = ground_truth[:, :, f]
        feat_mask = masks[:, :, f]

        # Denormalize
        pred_phys = denormalize(feat_pred, var)
        gt_phys   = denormalize(feat_gt, var)

        missing_idx = feat_mask == 1
        if missing_idx.sum() == 0:
            continue

        diff_phys = pred_phys[missing_idx] - gt_phys[missing_idx]
        mae_phys = np.abs(diff_phys).mean()
        rmse_phys = np.sqrt((diff_phys ** 2).mean())

        # Pearson correlation
        if len(missing_idx) > 1:
            pearson = np.corrcoef(pred_phys[missing_idx], gt_phys[missing_idx])[0, 1]
        else:
            pearson = 0.0

        metrics[f"mae_{VAR_NAMES[f]}"] = float(mae_phys)
        metrics[f"rmse_{VAR_NAMES[f]}"] = float(rmse_phys)
        metrics[f"pearson_{VAR_NAMES[f]}"] = float(pearson)

    return metrics


def eval_stat_baseline(name: str, func, data: np.ndarray,
                       mask_gen, mask_rate: float, device) -> dict:
    """Evaluate a statistical (no-training) baseline."""
    T, S, F = data.shape
    rng = np.random.RandomState(42)

    all_preds, all_gts, all_masks = [], [], []

    # Evaluate per-station on test split
    test_data = data[VAL_END - VAL_END: TEST_END - VAL_END]  # test portion
    test_T = test_data.shape[0]
    test_mask = mask_gen(test_T, F, mask_rate, seed=242)

    t0 = time.time()
    for s in range(min(S, 10)):  # Sample 10 stations for stat baselines
        gt = test_data[:, s, :]
        mask = test_mask.copy()
        observed = gt.copy()
        observed[mask.astype(bool)] = 0.0

        try:
            pred = func(observed, mask)
            if pred.ndim == 2:
                pred = pred[np.newaxis]
                gt_win = gt[np.newaxis]
                mask_win = mask[np.newaxis]
            # Take full test range as one big window
            all_preds.append(pred if pred.ndim == 3 else pred[np.newaxis])
            all_gts.append(gt[np.newaxis])
            all_masks.append(mask[np.newaxis])
        except Exception as e:
            print(f"    [WARN] {name} failed on station {s}: {e}")
            continue

    elapsed = time.time() - t0

    if not all_preds:
        return {"error": "all stations failed", "time_s": elapsed}

    preds = np.concatenate(all_preds, axis=0) if len(all_preds) > 1 else all_preds[0]
    gts   = np.concatenate(all_gts, axis=0) if len(all_gts) > 1 else all_gts[0]
    msks  = np.concatenate(all_masks, axis=0) if len(all_masks) > 1 else all_masks[0]

    metrics = compute_physical_metrics(preds, gts, msks)
    metrics["time_s"] = round(elapsed, 2)
    metrics["method"] = name
    return metrics


def eval_dl_model(model, loader: DataLoader, device, model_name: str,
                  is_saits: bool = False) -> dict:
    """Evaluate a trained DL model on a DataLoader."""
    model.eval()
    all_preds, all_gts, all_masks = [], [], []

    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            obs = batch["observed"].to(device)
            msk = batch["mask"].to(device)
            gt  = batch["ground_truth"].to(device)

            if is_saits:
                out = model(obs, msk)
                pred = out["output"]
            else:
                pred = model(obs, msk)

            all_preds.append(pred.cpu().numpy())
            all_gts.append(gt.cpu().numpy())
            all_masks.append(msk.cpu().numpy())

    elapsed = time.time() - t0

    preds = np.concatenate(all_preds, axis=0)
    gts   = np.concatenate(all_gts, axis=0)
    msks  = np.concatenate(all_masks, axis=0)

    metrics = compute_physical_metrics(preds, gts, msks)
    metrics["time_s"] = round(elapsed, 2)
    metrics["method"] = model_name
    return metrics


def build_dl_loader(data: np.ndarray, mask_gen, mask_rate: float,
                    batch_size: int, shuffle: bool = False) -> DataLoader:
    """Build a DataLoader for DL model evaluation (no stride, all windows)."""
    T, S, F = data.shape
    mask_tensor = torch.from_numpy(mask_gen(T, F, mask_rate, seed=242))
    ds = _SplitDataset(data, mask_tensor, WINDOW_SIZE, stride=1)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


# ── Experiment loops ────────────────────────────────────────────────────────

def run_baseline_experiments(data: np.ndarray) -> list:
    """Run statistical baselines across all patterns and rates."""
    results = []

    for pattern in MISSING_PATTERNS:
        mask_gen = MASK_GENERATORS[pattern]
        for rate in MISSING_RATES:
            print(f"\n{'─'*50}")
            print(f"  Baseline: pattern={pattern}, rate={rate:.0%}")
            print(f"{'─'*50}")

            for name, func in STAT_BASELINES.items():
                print(f"    {name}...", end=" ", flush=True)
                metrics = eval_stat_baseline(name, func, data, mask_gen, rate, DEVICE)
                metrics["pattern"] = pattern
                metrics["rate"] = rate
                results.append(metrics)
                if "error" in metrics:
                    print(f"  ✗ {metrics['error']}")
                else:
                    print(f"  MAE(phys)={metrics.get('mae_TEM', 'N/A'):.3f}")

            # KNN (special handling for speed — only one station)
            print(f"    knn...", end=" ", flush=True)
            try:
                metrics_knn = eval_stat_baseline(
                    "knn", lambda o, m: knn_imputation(o, m, k=5),
                    data, mask_gen, rate, DEVICE
                )
                metrics_knn["pattern"] = pattern
                metrics_knn["rate"] = rate
                results.append(metrics_knn)
                print(f"  MAE(phys)={metrics_knn.get('mae_TEM', 'N/A'):.3f}")
            except Exception as e:
                print(f"  ✗ {e}")

    return results


def run_dl_baseline_experiments(data: np.ndarray) -> list:
    """Train and evaluate DL baselines (BRITS, SAITS)."""
    results = []
    dl_baselines = {
        "brits": BRITS,
        "saits": SAITS,
    }

    for pattern in MISSING_PATTERNS:
        for rate in MISSING_RATES:
            print(f"\n{'─'*50}")
            print(f"  DL Baseline: pattern={pattern}, rate={rate:.0%}")
            print(f"{'─'*50}")

            # Build data loaders for this config
            T, S, F = data.shape
            mask_gen = MASK_GENERATORS[pattern]

            train_mask = torch.from_numpy(mask_gen(TRAIN_END, F, rate, seed=42))
            val_mask   = torch.from_numpy(mask_gen(VAL_END - TRAIN_END, F, rate, seed=142))
            test_mask  = torch.from_numpy(mask_gen(TEST_END - VAL_END, F, rate, seed=242))

            train_ds = _SplitDataset(data[:TRAIN_END], train_mask, WINDOW_SIZE, STRIDE)
            val_ds   = _SplitDataset(data[TRAIN_END:VAL_END], val_mask, WINDOW_SIZE, STRIDE)
            test_ds  = _SplitDataset(data[VAL_END:TEST_END], test_mask, WINDOW_SIZE, stride=1)

            dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=0,
                            pin_memory=torch.cuda.is_available())
            train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
            val_loader   = DataLoader(val_ds, shuffle=False, **dl_kwargs)
            test_loader  = DataLoader(test_ds, shuffle=False, **dl_kwargs)

            for name, model_cls in dl_baselines.items():
                print(f"    Training {name}...", flush=True)
                model = model_cls().to(DEVICE)
                is_saits = (name == "saits")

                config = {"epochs": EPOCHS, "lr": LR, "patience": PATIENCE}
                save_subdir = SAVE_DIR / "dl_baselines" / f"{pattern}_rate{int(rate*100)}"
                train_result = train_model(model, train_loader, val_loader,
                                          name, config, DEVICE, save_subdir)

                # Evaluate on test set
                test_metrics = eval_dl_model(model, test_loader, DEVICE, name, is_saits)
                test_metrics["pattern"] = pattern
                test_metrics["rate"] = rate
                test_metrics["train_result"] = train_result
                results.append(test_metrics)

                print(f"      Test MAE(phys)={test_metrics.get('mae_TEM', 'N/A'):.3f}")

    return results


def run_ablation_experiments(data: np.ndarray) -> list:
    """Train and evaluate MAB-Net ablation variants."""
    results = []
    model_names = ["mab_net", "bilstm_only", "transformer_only", "mab_net_no_mask"]

    for pattern in MISSING_PATTERNS:
        for rate in MISSING_RATES:
            print(f"\n{'─'*50}")
            print(f"  Ablation: pattern={pattern}, rate={rate:.0%}")
            print(f"{'─'*50}")

            T, S, F = data.shape
            mask_gen = MASK_GENERATORS[pattern]

            train_mask = torch.from_numpy(mask_gen(TRAIN_END, F, rate, seed=42))
            val_mask   = torch.from_numpy(mask_gen(VAL_END - TRAIN_END, F, rate, seed=142))
            test_mask  = torch.from_numpy(mask_gen(TEST_END - VAL_END, F, rate, seed=242))

            train_ds = _SplitDataset(data[:TRAIN_END], train_mask, WINDOW_SIZE, STRIDE)
            val_ds   = _SplitDataset(data[TRAIN_END:VAL_END], val_mask, WINDOW_SIZE, STRIDE)
            test_ds  = _SplitDataset(data[VAL_END:TEST_END], test_mask, WINDOW_SIZE, stride=1)

            dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=0,
                            pin_memory=torch.cuda.is_available())
            train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
            val_loader   = DataLoader(val_ds, shuffle=False, **dl_kwargs)
            test_loader  = DataLoader(test_ds, shuffle=False, **dl_kwargs)

            for name in model_names:
                print(f"    Training {name}...", flush=True)
                model = build_model(name).to(DEVICE)

                config = {"epochs": EPOCHS, "lr": LR, "patience": PATIENCE}
                save_subdir = SAVE_DIR / "ablation" / f"{pattern}_rate{int(rate*100)}"
                train_result = train_model(model, train_loader, val_loader,
                                          name, config, DEVICE, save_subdir)

                test_metrics = eval_dl_model(model, test_loader, DEVICE, name)
                test_metrics["pattern"] = pattern
                test_metrics["rate"] = rate
                test_metrics["train_result"] = train_result
                results.append(test_metrics)

                print(f"      Test MAE(phys)={test_metrics.get('mae_TEM', 'N/A'):.3f}")

    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MAB-Net Experiment Runner")
    parser.add_argument("--baseline-only", action="store_true", help="Run baselines only")
    parser.add_argument("--ablation-only", action="store_true", help="Run ablation only")
    parser.add_argument("--quick", action="store_true", help="Quick test (1 config)")
    parser.add_argument("--stat-only", action="store_true", help="Statistical baselines only")
    parser.add_argument("--device", type=str, default="auto", help="Device override")
    args = parser.parse_args()

    global DEVICE, MISSING_PATTERNS, MISSING_RATES, EPOCHS
    if args.device != "auto":
        DEVICE = torch.device(args.device)

    if args.quick:
        MISSING_PATTERNS = ["block"]
        MISSING_RATES = [0.80]
        EPOCHS = 5

    print("=" * 70)
    print("  MAB-Net Experiment Suite")
    print(f"  Device: {DEVICE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    matrices, station_info, _ = load_all_matrices(DATA_DIR)
    data = build_samples(matrices, split="all")  # (T, S, F) = (8784, 405, 3)
    print(f"  Data shape: {data.shape}  (T={data.shape[0]}, S={data.shape[1]}, F={data.shape[2]})")
    print(f"  Train: [0, {TRAIN_END})  Val: [{TRAIN_END}, {VAL_END})  Test: [{VAL_END}, {TEST_END})")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # JSON-safe type converter
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    if not args.ablation_only:
        # ── Baseline experiments ──
        print("\n[2/4] Running statistical baselines...")
        stat_results = run_baseline_experiments(data)
        all_results["stat_baselines"] = stat_results

    if args.stat_only:
        all_results["dl_baselines"] = []
        all_results["ablation"] = []
        summary_path = SAVE_DIR / "stat_results.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(convert(all_results), f, indent=2, ensure_ascii=False)
        print(f"\n{'='*70}")
        print(f"  ✅ Stat-only results saved to {summary_path}")
        print(f"{'='*70}")
        print_summary(all_results)
        return

    if not args.ablation_only:
        print("\n[3/4] Running DL baselines (BRITS, SAITS)...")
        dl_results = run_dl_baseline_experiments(data)
        all_results["dl_baselines"] = dl_results

    if not args.baseline_only:
        # ── Ablation experiments ──
        print("\n[4/4] Running ablation study...")
        ablation_results = run_ablation_experiments(data)
        all_results["ablation"] = ablation_results

    # ── Save summary ──
    summary_path = SAVE_DIR / "all_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(convert(all_results), f, indent=2, ensure_ascii=False)
    print(f"\n{'='*70}")
    print(f"  ✅ Results saved to {summary_path}")
    print(f"{'='*70}")

    # Print summary table
    print_summary(all_results)


def print_summary(all_results: dict):
    """Print a compact results table."""
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)

    # Headers
    header = f"{'Method':<20} {'Pattern':<12} {'Rate':>5}  {'MAE(TEM)':>10} {'RMSE(TEM)':>10} {'R(TEM)':>8} {'MAE(PRS)':>10} {'RMSE(PRS)':>10} {'R(PRS)':>8} {'MAE(WIN)':>10} {'RMSE(WIN)':>10} {'R(WIN)':>8}"
    print(header)
    print("-" * 80)

    def print_row(method, pattern, rate, m):
        if "error" in m:
            print(f"{method:<20} {pattern:<12} {rate:>4.0%}  ERROR: {m['error']}")
            return
        print(f"{method:<20} {pattern:<12} {rate:>4.0%}  "
              f"{m.get('mae_TEM', 0):>10.3f} {m.get('rmse_TEM', 0):>10.3f} {m.get('pearson_TEM', 0):>8.3f} "
              f"{m.get('mae_PRS', 0):>10.3f} {m.get('rmse_PRS', 0):>10.3f} {m.get('pearson_PRS', 0):>8.3f} "
              f"{m.get('mae_WIN', 0):>10.3f} {m.get('rmse_WIN', 0):>10.3f} {m.get('pearson_WIN', 0):>8.3f}")

    for category, results in all_results.items():
        if not results:
            continue
        print(f"\n  [{category}]")
        for r in results:
            print_row(r.get("method", "?"), r.get("pattern", "?"),
                     r.get("rate", 0), r)


if __name__ == "__main__":
    main()