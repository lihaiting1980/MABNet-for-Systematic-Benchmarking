"""
Phase A+B Experiment Runner: Window Sizes + Architecture Improvements
=====================================================================
Tests:
  A1: Window sizes 24 / 48 / 72 / 168h
  B1: Parallel vs Serial architecture
  B2: Smoothness regularization (β = 0, 0.05, 0.1, 0.2)

Focus: block missing (the hardest scenario), with spot-checks on continuous.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from dataloader import (
    load_all_matrices, _SplitDataset,
    TRAIN_END, VAL_END, TEST_END, MASK_GENERATORS,
)
from models import build_model, CompositeLoss
from models_v2 import (
    build_improved_model, MABNetParallel, MABNetDeepTransformer,
    SmoothCompositeLoss,
)
from baselines import STAT_BASELINES
from train import train_model, evaluate, CompositeLossDL
from run_cpu import compute_metrics, eval_stat

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(r"D:\tt\SCI\气象数据\processed_national_data")
SAVE_DIR = Path(r"D:\tt\SCI\experiments\results\phase_ab")
DEVICE = torch.device("cpu")
N_STATIONS = 10
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
PATIENCE = 8
STRIDE_MAP = {24: 24, 48: 48, 72: 72, 168: 168}  # 1 sample per window-length

# Focus on block missing (hardest), spot-check continuous
PRIMARY_PATTERNS = ["block"]
SPOT_PATTERNS = ["continuous"]
ALL_RATES = [0.30, 0.50, 0.80]

# Window sizes
WINDOW_SIZES = [24, 48, 72, 168]

# Model variants
SERIAL_MODELS = ["transformer_only"]  # best from v1 (as serial baseline)
PARALLEL_MODELS = ["mab_net_parallel"]
DEEP_MODELS = ["mab_net_deep"]

# Smoothness β values to test
BETA_VALUES = [0.0, 0.05, 0.1, 0.2]

with open(DATA_DIR / "standardization_params.json") as f:
    STD_PARAMS = json.load(f)

print(f"Phase A+B: Window sizes + Architecture improvements")
print(f"Device: {DEVICE} | Stations: {N_STATIONS} | Epochs: {EPOCHS}")


# ── Data loading helpers ────────────────────────────────────────────────────

def build_loaders_for_window(data, mask_gen, rate, window_size, stride, shuffle=True):
    """Build train/val/test loaders with a specific window size."""
    T, S, F = data.shape
    train_mask = torch.from_numpy(mask_gen(TRAIN_END, F, rate, seed=42))
    val_mask   = torch.from_numpy(mask_gen(VAL_END - TRAIN_END, F, rate, seed=142))
    test_mask  = torch.from_numpy(mask_gen(TEST_END - VAL_END, F, rate, seed=242))

    sel = slice(0, N_STATIONS)
    train_ds = _SplitDataset(data[:TRAIN_END, sel, :], train_mask, window_size, stride)
    val_ds   = _SplitDataset(data[TRAIN_END:VAL_END, sel, :], val_mask, window_size, stride)
    test_ds  = _SplitDataset(data[VAL_END:TEST_END, sel, :], test_mask, window_size, stride=1)

    dl_kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=shuffle, **dl_kw),
        DataLoader(val_ds, shuffle=False, **dl_kw),
        DataLoader(test_ds, shuffle=False, **dl_kw),
    )


def eval_dl_full(model, loader, is_v2=False):
    """Evaluate a trained DL model, extracting all predictions."""
    model.eval()
    all_preds, all_gts, all_masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            obs = batch["observed"].to(DEVICE)
            msk = batch["mask"].to(DEVICE)
            gt  = batch["ground_truth"].to(DEVICE)

            if is_v2:
                pred = model(obs, msk)
            else:
                pred = model(obs, msk)

            all_preds.append(pred.cpu().numpy())
            all_gts.append(gt.cpu().numpy())
            all_masks.append(msk.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    gts   = np.concatenate(all_gts, axis=0)
    msks  = np.concatenate(all_masks, axis=0)
    return compute_metrics(preds, gts, msks)


# ── Training with smoothness loss ───────────────────────────────────────────

def train_with_smoothness(model, train_loader, val_loader, model_name,
                          config, device, save_dir, beta=0.1):
    """Training loop with smoothness-regularized loss."""
    epochs = config.get("epochs", 20)
    lr = config.get("lr", 1e-3)
    patience = config.get("patience", 8)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = SmoothCompositeLoss(alpha=1.0, beta=beta)

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
            optimizer.zero_grad()
            pred = model(obs, msk)
            loss = criterion(pred, gt, msk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch["observed"].to(device)
                msk = batch["mask"].to(device)
                gt  = batch["ground_truth"].to(device)
                pred = model(obs, msk)
                val_loss += criterion(pred, gt, msk).item()
        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)

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


# ── Main experiment ─────────────────────────────────────────────────────────

def main():
    t_total = time.time()
    print("=" * 60)
    print("  Phase A+B: Window Sizes + Architecture Improvements")
    print("=" * 60)

    # Load data
    matrices, _, _ = load_all_matrices(DATA_DIR)
    data = np.stack([
        matrices["temperature_std"],
        matrices["pressure_std"],
        matrices["wind_speed_std"],
    ], axis=-1)
    print(f"  Data: {data.shape}")

    all_results = {
        "phase": "A+B",
        "config": {"n_stations": N_STATIONS, "epochs": EPOCHS},
        "experiments": [],
    }

    run_id = 0
    total_runs = (len(PRIMARY_PATTERNS) * len(ALL_RATES) *
                  (len(WINDOW_SIZES) * len(SERIAL_MODELS + PARALLEL_MODELS) +
                   len(DEEP_MODELS) + len(BETA_VALUES)))

    for pattern in PRIMARY_PATTERNS:
        mask_gen = MASK_GENERATORS[pattern]

        for rate in ALL_RATES:
            print(f"\n{'#'*55}")
            print(f"#  [{pattern} @ {rate:.0%}]")
            print(f"{'#'*55}")

            # ── A1: Window size ablation ──
            print("\n  ── A1: Window Size Ablation ──")
            for ws in WINDOW_SIZES:
                stride = STRIDE_MAP[ws]
                train_l, val_l, test_l = build_loaders_for_window(
                    data, mask_gen, rate, ws, stride
                )
                n_train = len(train_l.dataset)
                print(f"\n  Window={ws}h (stride={stride}h, train_samples={n_train})")

                for arch_name in SERIAL_MODELS + PARALLEL_MODELS:
                    run_id += 1
                    is_v2 = arch_name in PARALLEL_MODELS or arch_name in DEEP_MODELS
                    print(f"    [{run_id}] {arch_name}...", end=" ", flush=True)

                    t1 = time.time()
                    if arch_name in PARALLEL_MODELS + DEEP_MODELS:
                        model = build_improved_model(arch_name).to(DEVICE)
                    else:
                        model = build_model(arch_name).to(DEVICE)

                    config = {"epochs": EPOCHS, "lr": LR, "patience": PATIENCE}
                    subdir = SAVE_DIR / f"window_{ws}" / f"{pattern}_r{int(rate*100)}"
                    subdir.mkdir(parents=True, exist_ok=True)

                    train_result = train_model(model, train_l, val_l,
                                              arch_name, config, DEVICE, subdir)
                    test_m = eval_dl_full(model, test_l, is_v2=is_v2)

                    test_m["method"] = arch_name
                    test_m["window_size"] = ws
                    test_m["pattern"] = pattern
                    test_m["rate"] = rate
                    test_m["experiment"] = "A1_window"
                    test_m["time_s"] = round(time.time() - t1, 1)
                    all_results["experiments"].append(test_m)

                    print(f"MAE(TEM)={test_m.get('mae_TEM', -1):.2f}°C "
                          f"({test_m['time_s']}s)")

            # ── B2: Smoothness regularization ──
            # Use best-performing window from A1 (default to 168h for block)
            best_ws = 168 if pattern == "block" else 72
            stride = STRIDE_MAP[best_ws]
            print(f"\n  ── B2: Smoothness Regularization (window={best_ws}h) ──")

            for beta in BETA_VALUES:
                run_id += 1
                print(f"    [{run_id}] mab_net_parallel β={beta}...", end=" ", flush=True)

                t1 = time.time()
                train_l, val_l, test_l = build_loaders_for_window(
                    data, mask_gen, rate, best_ws, stride
                )

                model = build_improved_model("mab_net_parallel").to(DEVICE)
                config = {"epochs": EPOCHS, "lr": LR, "patience": PATIENCE}
                subdir = SAVE_DIR / f"smoothness_b{int(beta*100)}"
                subdir.mkdir(parents=True, exist_ok=True)

                train_with_smoothness(model, train_l, val_l, "mab_net_parallel",
                                     config, DEVICE, subdir, beta=beta)
                test_m = eval_dl_full(model, test_l, is_v2=True)

                test_m["method"] = f"mab_net_parallel_β={beta}"
                test_m["window_size"] = best_ws
                test_m["pattern"] = pattern
                test_m["rate"] = rate
                test_m["experiment"] = "B2_smoothness"
                test_m["time_s"] = round(time.time() - t1, 1)
                all_results["experiments"].append(test_m)

                print(f"MAE(TEM)={test_m.get('mae_TEM', -1):.2f}°C")

            # ── B3: Deep Transformer ──
            for ws in [best_ws]:
                stride = STRIDE_MAP[ws]
                train_l, val_l, test_l = build_loaders_for_window(
                    data, mask_gen, rate, ws, stride
                )
                for arch_name in DEEP_MODELS:
                    run_id += 1
                    print(f"    [{run_id}] {arch_name}(ws={ws}h)...",
                          end=" ", flush=True)
                    t1 = time.time()
                    model = build_improved_model(arch_name).to(DEVICE)
                    config = {"epochs": EPOCHS, "lr": LR, "patience": PATIENCE}
                    subdir = SAVE_DIR / f"deep_{ws}"
                    subdir.mkdir(parents=True, exist_ok=True)

                    train_result = train_model(model, train_l, val_l,
                                              arch_name, config, DEVICE, subdir)
                    test_m = eval_dl_full(model, test_l, is_v2=True)
                    test_m["method"] = arch_name
                    test_m["window_size"] = ws
                    test_m["pattern"] = pattern
                    test_m["rate"] = rate
                    test_m["experiment"] = "B3_deep"
                    test_m["time_s"] = round(time.time() - t1, 1)
                    all_results["experiments"].append(test_m)

                    print(f"MAE(TEM)={test_m.get('mae_TEM', -1):.2f}°C")

            # Save checkpoint
            with open(SAVE_DIR / "phase_ab_partial.json", "w") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # ── Final save ──
    total_min = (time.time() - t_total) / 60
    final_path = SAVE_DIR / "phase_ab_results.json"
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*60}")
    print(f"  ✅ Phase A+B done! Total: {total_min:.0f} min")
    print(f"  Results: {final_path}")
    print_summary(all_results)


def print_summary(results):
    """Print results grouped by experiment type."""
    exps = results.get("experiments", [])
    if not exps:
        return

    print("\n" + "=" * 90)
    print("  PHASE A+B RESULTS")
    print("=" * 90)

    # Group by experiment
    for exp_type in ["A1_window", "B2_smoothness", "B3_deep"]:
        group = [e for e in exps if e.get("experiment") == exp_type]
        if not group:
            continue
        print(f"\n  [{exp_type}]")
        print(f"  {'Method':<30} {'Win':>4} {'Pat':<10} {'Rate':>4}  "
              f"{'TEM':>7} {'PRS':>7} {'WIN':>7} {'R(T)':>6} {'Time':>6}")
        print("  " + "-" * 88)
        for r in sorted(group, key=lambda x: (x.get("rate", 0),
                                               x.get("mae_TEM", 99))):
            print(f"  {r.get('method','?'):<30} {r.get('window_size','-'):>4} "
                  f"{r.get('pattern',''):<10} {r.get('rate',0):>3.0%}  "
                  f"{r.get('mae_TEM',0):>7.2f} {r.get('mae_PRS',0):>7.2f} "
                  f"{r.get('mae_WIN',0):>7.2f} {r.get('pearson_TEM',0):>6.3f} "
                  f"{r.get('time_s','-'):>5}")


if __name__ == "__main__":
    main()