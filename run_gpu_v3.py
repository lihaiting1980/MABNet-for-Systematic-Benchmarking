"""
Full-Scale GPU Experiment Runner
=================================
Runs on AutoDL RTX 3090 with all 405 stations.

Tests:
  Main:   mab_net_parallel + spatial + temporal (168h)
          3 patterns × 3 rates × 3 seeds = 27 runs
  Ablation: 4 variants × block × 3 rates = 12 runs
  Baselines: 6 methods × 3 patterns × 3 rates = 9 stat evals

Total: 48 experiment configs
"""

import json, time, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/root/experiments")
from dataloader import (
    load_all_matrices, _SplitDataset, TRAIN_END, VAL_END, TEST_END,
    WINDOW_SIZE, MASK_GENERATORS,
)
from models_v2 import build_improved_model
from baselines import STAT_BASELINES

# ── Config ──
DATA_DIR = Path("/root/data")
SAVE_DIR = Path("/root/results_v3")
DEVICE = torch.device("cuda")
WINDOW = 336        # best from Phase A
STRIDE = 336          # denser sampling on GPU
N_STATIONS = 20    # ALL stations
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
PATIENCE = 5
WARMUP_EPOCHS = 3
SEEDS = [42]

PATTERNS = ["random", "continuous", "block"]
RATES = [0.30, 0.50, 0.80]

SAVE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE} | Window: {WINDOW}h | Stride: {STRIDE} | Stations: {N_STATIONS}")
print(f"Batch: {BATCH_SIZE} | Epochs: {EPOCHS} | Seeds: {SEEDS}")

# ── Load data ──
print("\nLoading data...")
matrices, _, coords = load_all_matrices(DATA_DIR)
data = np.stack([matrices["temperature_std"], matrices["pressure_std"],
                 matrices["wind_speed_std"]], axis=-1)
print(f"Data: {data.shape}")

# Station coordinates
station_lats = torch.tensor(coords["latitude"].values[:N_STATIONS].astype(np.float32))
station_lons = torch.tensor(coords["longitude"].values[:N_STATIONS].astype(np.float32))


# ── Metrics ──
with open(DATA_DIR / "standardization_params.json") as f:
    STD_PARAMS = json.load(f)

def compute_metrics(pred, gt, mask):
    """MAE, RMSE, Pearson in physical units."""
    var_map = {0: "temperature", 1: "pressure", 2: "wind_speed"}
    out = {}
    diff = pred - gt
    out["mae_std"] = float(np.abs(diff).mean())
    out["rmse_std"] = float(np.sqrt((diff**2).mean()))
    for f, label in enumerate(["TEM", "PRS", "WIN"]):
        mf = mask[:, :, f] == 1
        if mf.sum() < 2: continue
        v = var_map[f]
        pp = pred[:, :, f][mf] * STD_PARAMS[v]["std"] + STD_PARAMS[v]["mean"]
        gg = gt[:, :, f][mf] * STD_PARAMS[v]["std"] + STD_PARAMS[v]["mean"]
        out[f"mae_{label}"] = float(np.abs(pp - gg).mean())
        out[f"rmse_{label}"] = float(np.sqrt(((pp - gg)**2).mean()))
        out[f"pearson_{label}"] = float(np.corrcoef(pp, gg)[0,1]) if len(pp)>1 else 0
    return out


# ── Dataset ──
from torch.utils.data import Dataset

class FullDataset(Dataset):
    def __init__(self, data, mask_tensor, window, stride, lats, lons):
        self.data = torch.from_numpy(data)
        self.mask = mask_tensor
        self.window = window
        self.T, self.S, self.F = data.shape
        self.lats = lats
        self.lons = lons
        self.indices = []
        for s in range(self.S):
            for t in range(0, self.T - window + 1, stride):
                self.indices.append((s, t))
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        s, t = self.indices[idx]; e = t + self.window
        gt = self.data[t:e, s, :]
        m = self.mask[t:e, :]
        obs = gt.clone(); obs[m.bool()] = 0.0
        return {"observed": obs, "mask": m, "ground_truth": gt,
                "lat": self.lats[s], "lon": self.lons[s],
                "timestamps": torch.arange(t, e, dtype=torch.float32)}


def build_loaders(data, mask_gen, rate, window, stride, seed, shuffle=True):
    T, S, F = data.shape
    rng = np.random.RandomState(seed)
    train_mask = torch.from_numpy(mask_gen(TRAIN_END, F, rate, seed=seed))
    val_mask   = torch.from_numpy(mask_gen(VAL_END-TRAIN_END, F, rate, seed=seed+1000))
    test_mask  = torch.from_numpy(mask_gen(TEST_END-VAL_END, F, rate, seed=seed+2000))

    train_ds = FullDataset(data[:TRAIN_END], train_mask, window, stride, station_lats, station_lons)
    val_ds   = FullDataset(data[TRAIN_END:VAL_END], val_mask, window, stride, station_lats, station_lons)
    test_ds  = FullDataset(data[VAL_END:TEST_END], test_mask, window, stride=1, lats=station_lats, lons=station_lons)

    kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    return (DataLoader(train_ds, shuffle=shuffle, **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw))


# ── Training ──
def train_gpu(model, train_l, val_l, epochs, lr, patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = torch.nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    pcount = 0

    for ep in range(1, epochs+1):
        model.train(); tl = 0.0
        for b in train_l:
            obs, msk, gt = b["observed"].cuda(), b["mask"].cuda(), b["ground_truth"].cuda()
            lat, lon, ts = b["lat"].cuda(), b["lon"].cuda(), b["timestamps"].cuda()
            optimizer.zero_grad()
            pred = model(obs, msk, lat=lat, lon=lon, timestamps=ts)
            loss = (criterion(pred, gt) * msk).sum() / (msk.sum()+1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()
        tl /= max(len(train_l), 1)

        model.eval(); vl = 0.0
        with torch.no_grad():
            for b in val_l:
                obs, msk, gt = b["observed"].cuda(), b["mask"].cuda(), b["ground_truth"].cuda()
                lat, lon, ts = b["lat"].cuda(), b["lon"].cuda(), b["timestamps"].cuda()
                pred = model(obs, msk, lat=lat, lon=lon, timestamps=ts)
                vl += ((criterion(pred, gt)*msk).sum()/(msk.sum()+1e-8)).item()
        vl /= max(len(val_l), 1)
        scheduler.step(torch.tensor(vl))

        if vl < best_loss:
            best_loss = vl; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            pcount = 0
        else: pcount += 1
        if pcount >= patience: break

    model.load_state_dict(best_state)
    return best_loss

def eval_gpu(model, loader):
    model.eval()
    preds, gts, msks = [], [], []
    with torch.no_grad():
        for b in loader:
            obs, msk, gt = b["observed"].cuda(), b["mask"].cuda(), b["ground_truth"].cuda()
            lat, lon, ts = b["lat"].cuda(), b["lon"].cuda(), b["timestamps"].cuda()
            pred = model(obs, msk, lat=lat, lon=lon, timestamps=ts)
            preds.append(pred.cpu().numpy()); gts.append(gt.cpu().numpy()); msks.append(msk.cpu().numpy())
    return compute_metrics(np.concatenate(preds), np.concatenate(gts), np.concatenate(msks))


# ── Statistical baselines ──
def eval_stat_baselines(data_test, mask_gen, rate, seed):
    T, S, F = data_test.shape
    test_mask = mask_gen(T, F, rate, seed=seed+2000)
    results = {}
    for name, func in STAT_BASELINES.items():
        all_p, all_g, all_m = [], [], []
        for s in range(0, S, 5):  # sample every 5th station
            gt = data_test[:, s, :]; obs = gt.copy(); obs[test_mask.astype(bool)] = 0.0
            try:
                p = func(obs, test_mask)
                all_p.append(p[np.newaxis]); all_g.append(gt[np.newaxis]); all_m.append(test_mask[np.newaxis])
            except: continue
        if all_p:
            results[name] = compute_metrics(np.concatenate(all_p), np.concatenate(all_g), np.concatenate(all_m))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
all_results = {"config": {"window": WINDOW, "stride": STRIDE, "stations": N_STATIONS,
                          "epochs": EPOCHS, "seeds": SEEDS, "device": str(DEVICE)},
               "main": [], "ablation": [], "baselines": []}

run_id = 0
total = len(PATTERNS) * len(RATES) * len(SEEDS) + 12 + 9  # 27+12+9=48

print(f"\n{'='*60}\n  STARTING {total} EXPERIMENTS\n{'='*60}")

# ── MAIN EXPERIMENTS ──
for pattern in PATTERNS:
    mask_gen = MASK_GENERATORS[pattern]
    for rate in RATES:
        for seed in SEEDS:
            run_id += 1
            print(f"\n[{run_id}/{total}] MAIN: {pattern} {rate:.0%} seed={seed}")
            t1 = time.time()

            train_l, val_l, test_l = build_loaders(data, mask_gen, rate, WINDOW, STRIDE, seed)
            model = build_improved_model("mab_net_parallel", use_spatial=True, use_temporal=True).cuda()

            train_gpu(model, train_l, val_l, EPOCHS, LR, PATIENCE)
            m = eval_gpu(model, test_l)
            m.update({"pattern": pattern, "rate": rate, "seed": seed, "method": "mab_net_parallel+spatial+temp",
                       "time_s": round(time.time()-t1, 1)})
            all_results["main"].append(m)
            print(f"  MAE(TEM)={m.get('mae_TEM',0):.2f}C, R={m.get('pearson_TEM',0):.3f}, {m['time_s']}s")

            # Save checkpoint every 3 runs
            if run_id % 3 == 0:
                with open(SAVE_DIR / "results_partial.json", "w") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

# ── ABLATION ──
ablation_models = {
    "bilstm_only":       {"use_spatial": False, "use_temporal": False},
    "transformer_only":  {"use_spatial": False, "use_temporal": False},
    "mab_net_no_mask":   {"use_spatial": False, "use_temporal": False},
    "mab_net_parallel_nocoding": {"use_spatial": False, "use_temporal": False},
}

for pattern in ["block"]:  # hardest scenario for ablation
    mask_gen = MASK_GENERATORS[pattern]
    for rate in RATES:
        for name, kwargs in ablation_models.items():
            run_id += 1
            print(f"\n[{run_id}/{total}] ABLATION: {name} {pattern} {rate:.0%}")
            t1 = time.time()

            seed = 42
            train_l, val_l, test_l = build_loaders(data, mask_gen, rate, WINDOW, STRIDE, seed)

            if name == "bilstm_only":
                from models import BiLSTMOnly
                model = BiLSTMOnly().cuda()
                is_ablation_special = True
            elif name == "transformer_only":
                from models import TransformerOnly
                model = TransformerOnly().cuda()
                is_ablation_special = True
            elif name == "mab_net_no_mask":
                from models import MABNetNoMask
                model = MABNetNoMask().cuda()
                is_ablation_special = True
            else:
                model = build_improved_model("mab_net_parallel", **kwargs).cuda()
                is_ablation_special = False

            if is_ablation_special:
                # Use standard training (no spatial/temporal args)
                from train import train_model, evaluate, CompositeLossDL
                # Build standard dataset
                train_mask = torch.from_numpy(mask_gen(TRAIN_END, 3, rate, seed=seed))
                val_mask = torch.from_numpy(mask_gen(VAL_END-TRAIN_END, 3, rate, seed=seed+1000))
                test_mask = torch.from_numpy(mask_gen(TEST_END-VAL_END, 3, rate, seed=seed+2000))
                train_ds = _SplitDataset(data[:TRAIN_END], train_mask, WINDOW, STRIDE)
                val_ds = _SplitDataset(data[TRAIN_END:VAL_END], val_mask, WINDOW, STRIDE)
                test_ds = _SplitDataset(data[VAL_END:TEST_END], test_mask, WINDOW, stride=1)
                kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
                std_train_l = DataLoader(train_ds, shuffle=True, **kw)
                std_val_l = DataLoader(val_ds, shuffle=False, **kw)
                std_test_l = DataLoader(test_ds, shuffle=False, **kw)
                train_model(model, std_train_l, std_val_l, name, {"epochs": EPOCHS, "lr": LR, "patience": PATIENCE},
                           DEVICE, SAVE_DIR / "ckpt")
                # Evaluate
                model.eval()
                preds, gts, msks = [], [], []
                with torch.no_grad():
                    for b in std_test_l:
                        obs, msk, gt = b["observed"].cuda(), b["mask"].cuda(), b["ground_truth"].cuda()
                        pred = model(obs, msk)
                        preds.append(pred.cpu().numpy()); gts.append(gt.cpu().numpy()); msks.append(msk.cpu().numpy())
                m = compute_metrics(np.concatenate(preds), np.concatenate(gts), np.concatenate(msks))
            else:
                train_gpu(model, train_l, val_l, EPOCHS, LR, PATIENCE)
                m = eval_gpu(model, test_l)

            m.update({"pattern": pattern, "rate": rate, "method": name,
                       "time_s": round(time.time()-t1, 1)})
            all_results["ablation"].append(m)
            print(f"  MAE(TEM)={m.get('mae_TEM',0):.2f}C, R={m.get('pearson_TEM',0):.3f}")

    with open(SAVE_DIR / "results_partial.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

# ── STATISTICAL BASELINES ──
data_test = data[VAL_END-VAL_END:TEST_END-VAL_END]
for pattern in PATTERNS:
    mask_gen = MASK_GENERATORS[pattern]
    for rate in RATES:
        run_id += 1
        print(f"\n[{run_id}/{total}] BASELINES: {pattern} {rate:.0%}")
        m = eval_stat_baselines(data_test, mask_gen, rate, 42)
        for name, metrics in m.items():
            metrics.update({"pattern": pattern, "rate": rate, "method": name})
            all_results["baselines"].append(metrics)

# ── SAVE ──
final_path = SAVE_DIR / "full_gpu_results.json"
with open(final_path, "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

elapsed = (time.time() - t0) / 60
print(f"\n{'='*60}")
print(f"  DONE! {total} experiments in {elapsed:.0f} min")
print(f"  Results: {final_path}")
print(f"{'='*60}")
print_summary(all_results)


def print_summary(res):
    print("\n" + "="*70 + "\n  SUMMARY\n" + "="*70)
    for cat, items in [("MAIN", res["main"]), ("ABLATION", res["ablation"])]:
        if not items: continue
        print(f"\n[{cat}]")
        print(f"  {'Method':<35} {'Pat':<10} {'Rate':>4} {'MAE_T':>7} {'R_T':>6} {'MAE_P':>7} {'MAE_W':>7}")
        for r in sorted(items, key=lambda x: (x.get("rate",0), x.get("mae_TEM",99))):
            print(f"  {r.get('method','?'):<35} {r.get('pattern',''):<10} {r.get('rate',0):>3.0%}  "
                  f"{r.get('mae_TEM',0):>7.2f} {r.get('pearson_TEM',0):>6.3f} "
                  f"{r.get('mae_PRS',0):>7.2f} {r.get('mae_WIN',0):>7.2f}")