"""
Training Loop
=============
Generic training for all DL models (MAB-Net variants, BRITS, SAITS).
Handles early stopping, learning rate scheduling, and logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
from pathlib import Path
import json
import time


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion, device: torch.device, model_name: str = "",
                is_saits: bool = False) -> float:
    """Single training epoch.  Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        obs = batch["observed"].to(device)
        msk = batch["mask"].to(device)
        gt  = batch["ground_truth"].to(device)

        optimizer.zero_grad()

        if is_saits:
            out = model(obs, msk)
            pred = out["output"]
            # SAITS dual loss: imputation + reconstruction
            imp_loss = criterion(pred, gt, msk)
            recon_loss = criterion(out["reconstructed"], obs, 1 - msk)
            loss = imp_loss + 0.1 * recon_loss
        else:
            pred = model(obs, msk)
            loss = criterion(pred, gt, msk)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion,
             device: torch.device, model_name: str = "",
             is_saits: bool = False) -> Dict[str, float]:
    """Evaluate model on a dataset.  Returns loss and per-variable metrics.

    Returns dict with keys: loss, mae_0, mae_1, mae_2, rmse_0, rmse_1, rmse_2
    where 0=TEM, 1=PRS, 2=WIN.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    # Accumulators for per-variable metrics
    mae_sum = torch.zeros(3, device=device)
    mse_sum = torch.zeros(3, device=device)
    n_missing_feat = torch.zeros(3, device=device)

    for batch in loader:
        obs = batch["observed"].to(device)
        msk = batch["mask"].to(device)
        gt  = batch["ground_truth"].to(device)

        if is_saits:
            out = model(obs, msk)
            pred = out["output"]
        else:
            pred = model(obs, msk)

        loss = criterion(pred, gt, msk)
        total_loss += loss.item()
        n_batches += 1

        # Per-variable metrics
        diff = pred - gt
        abs_diff = diff.abs()
        sq_diff = diff ** 2
        for f in range(3):
            feat_mask = msk[:, :, f]  # (B, T)
            n_f = feat_mask.sum()
            if n_f > 0:
                mae_sum[f] += (abs_diff[:, :, f] * feat_mask).sum()
                mse_sum[f] += (sq_diff[:, :, f] * feat_mask).sum()
                n_missing_feat[f] += n_f

    metrics = {"loss": total_loss / max(n_batches, 1)}
    for f, name in enumerate(["TEM", "PRS", "WIN"]):
        n = max(n_missing_feat[f], 1)
        metrics[f"mae_{name}"] = (mae_sum[f] / n).item()
        metrics[f"rmse_{name}"] = torch.sqrt(mse_sum[f] / n).item()

    # Overall MAE & RMSE
    total_n = max(n_missing_feat.sum(), 1)
    metrics["mae"]  = (mae_sum.sum() / total_n).item()
    metrics["rmse"] = torch.sqrt(mse_sum.sum() / total_n).item()

    return metrics


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                model_name: str, config: dict, device: torch.device,
                save_dir: Path) -> Dict:
    """Full training loop with early stopping.

    Returns best validation metrics dict.
    """
    epochs = config.get("epochs", 50)
    lr = config.get("lr", 1e-3)
    patience = config.get("patience", 10)
    is_saits = model_name == "saits"

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = CompositeLossDL(alpha=1.0)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    print(f"\n{'='*60}")
    print(f"Training {model_name} | epochs={epochs} | lr={lr} | patience={patience}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 device, model_name, is_saits)
        val_metrics = evaluate(model, val_loader, criterion, device,
                               model_name, is_saits)
        val_loss = val_metrics["loss"]
        elapsed = time.time() - t0

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                  f"val_MAE={val_metrics['mae']:.4f} | "
                  f"best_epoch={best_epoch} | {elapsed:.1f}s")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best
    model.load_state_dict(best_state)
    print(f"  ✅ Best epoch: {best_epoch}, val_loss={best_val_loss:.4f}")

    # Save checkpoint
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_name": model_name,
        "state_dict": best_state,
        "config": config,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }, save_dir / f"{model_name}.pt")

    return {"best_epoch": best_epoch, "best_val_loss": best_val_loss,
            "val_metrics": history["val_metrics"][best_epoch - 1],
            "history": history}


class CompositeLossDL(nn.Module):
    """Same as in models.py — duplicated to keep training.py self-contained."""
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction="none")
        self.mae = nn.L1Loss(reduction="none")

    def forward(self, pred, target, mask):
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        combined = mse_loss + self.alpha * mae_loss
        return (combined * mask).sum() / (mask.sum() + 1e-8)