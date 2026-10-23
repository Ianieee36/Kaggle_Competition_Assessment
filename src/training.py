"""Training, evaluation and prediction functions (report Tasks 3, 4 and 5).

The notebook calls these functions, so the training loop is defined once and every run
(baseline CNN, frozen ResNet-50, fine-tuned ResNet-50) is trained in exactly the same way.

Per epoch, the loop records training and validation loss and accuracy plus the learning rate,
saves the history to CSV (for learning curves), keeps the weights from the epoch with the best
validation accuracy, and stops early if validation loss stops improving.
"""

import random
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
from torch import nn

import preprocessing as pp
from models import build_model, count_parameters

MODEL_DIR = pp.OUTPUT_DIR / "models"
HISTORY_DIR = pp.OUTPUT_DIR / "history"


@dataclass
class RunConfig:
    """All settings for one training run, so each run is fully described in one place."""
    name: str                     # run name, used for saved files
    model: str                    # 'baseline_cnn', 'resnet50_frozen' or 'resnet50_finetune'
    lr: float                     # starting learning rate
    epochs: int                   # maximum number of epochs (early stopping may end sooner)
    weight_decay: float = 1e-4    # L2-style regularisation applied by AdamW
    label_smoothing: float = 0.1  # softens targets to reduce overconfidence
    patience: int = 4             # epochs without validation-loss improvement before stopping
    batch_size: int = 64
    num_workers: int = 4
    init_from: str | None = None  # optional checkpoint to start from (e.g. the frozen run)


def set_seed(seed=pp.SEED):
    """Seeds every random number generator used, so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# One epoch of training / evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_outputs=False):
    """Loss and accuracy on a labelled loader. With return_outputs=True, also returns
    true labels, predicted labels and softmax probabilities for detailed evaluation."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_y, all_prob = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
        if return_outputs:
            all_y.append(y.cpu())
            all_prob.append(torch.softmax(logits, 1).cpu())
    if not return_outputs:
        return total_loss / n, correct / n
    y_true = torch.cat(all_y).numpy()
    probs = torch.cat(all_prob).numpy()
    return total_loss / n, correct / n, y_true, probs.argmax(1), probs


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------
def fit(cfg: RunConfig, device, train_loader, val_loader, pretrained=True):
    """Trains one run and returns (model with best weights loaded, history DataFrame)."""
    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg.model, pretrained=pretrained)
    if cfg.init_from:
        state = torch.load(MODEL_DIR / f"{cfg.init_from}.pt", map_location="cpu")
        model.load_state_dict(state)
        print(f"Initialised from checkpoint: {cfg.init_from}")
    model.to(device)

    total, trainable = count_parameters(model)
    print(f"[{cfg.name}] parameters: {total:,} total, {trainable:,} trainable | device: {device}")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Cosine schedule: the learning rate falls smoothly from lr towards zero over the planned epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    history, best_acc, best_loss, stale = [], -1.0, float("inf"), 0
    run_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        improved_acc = va_acc > best_acc
        if improved_acc:                                  # keep the weights with the best validation accuracy
            best_acc = va_acc
            torch.save(model.state_dict(), MODEL_DIR / f"{cfg.name}.pt")
        if va_loss < best_loss - 1e-4:                    # early stopping watches validation loss
            best_loss, stale = va_loss, 0
        else:
            stale += 1

        secs = time.time() - start
        history.append(dict(epoch=epoch, lr=lr, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, seconds=secs))
        pd.DataFrame(history).to_csv(HISTORY_DIR / f"{cfg.name}.csv", index=False)
        print(f"  epoch {epoch:2d} | lr {lr:.2e} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {secs:.0f}s" + ("  *best" if improved_acc else ""))

        if stale >= cfg.patience:
            print(f"  Early stopping: validation loss has not improved for {cfg.patience} epochs.")
            break

    model.load_state_dict(torch.load(MODEL_DIR / f"{cfg.name}.pt", map_location=device))
    hist = pd.DataFrame(history)
    best = hist.loc[hist["val_acc"].idxmax()]
    print(f"[{cfg.name}] best val acc {best['val_acc']:.4f} at epoch {int(best['epoch'])} | "
          f"total time {(time.time() - run_start) / 60:.1f} min")
    return model, hist


def train_or_load(cfg: RunConfig, device, train_loader, val_loader, retrain=False):
    """Trains a run, or loads its saved weights and history if it has already been trained.
    This lets the notebook be re-run (e.g. after a kernel restart) without repeating long runs.
    Set retrain=True to force a fresh training run."""
    ckpt, hist_csv = MODEL_DIR / f"{cfg.name}.pt", HISTORY_DIR / f"{cfg.name}.csv"
    if ckpt.exists() and hist_csv.exists() and not retrain:
        model = load_trained_model(cfg.name, cfg.model, device)
        hist = pd.read_csv(hist_csv)
        best = hist.loc[hist["val_acc"].idxmax()]
        print(f"[{cfg.name}] loaded saved run: {len(hist)} epochs, best val acc {best['val_acc']:.4f} "
              f"at epoch {int(best['epoch'])} (set retrain=True to train again)")
        return model, hist
    return fit(cfg, device, train_loader, val_loader)


def load_trained_model(name, model_type, device):
    """Loads saved weights for evaluation or prediction (no download of ImageNet weights needed)."""
    model = build_model(model_type, pretrained=False)
    model.load_state_dict(torch.load(MODEL_DIR / f"{name}.pt", map_location="cpu"))
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Pipeline sanity check
# ---------------------------------------------------------------------------
def overfit_one_batch(model, x, y, device, steps=60, lr=1e-3):
    """Trains repeatedly on a single batch. If the model, loss and optimiser are wired correctly,
    the loss should fall close to zero. Returns the loss at each step."""
    model = model.to(device).train()
    x, y = x.to(device), y.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    acc = (model(x).argmax(1) == y).float().mean().item()
    return losses, acc


# ---------------------------------------------------------------------------
# Prediction and submission
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device, max_batches=None):
    """Predicts test images. Returns a DataFrame with image_id, predicted class name and confidence,
    in the loader's order (which follows sample_submission.csv)."""
    model.eval()
    ids, preds, confs = [], [], []
    for i, (x, batch_ids) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        probs = torch.softmax(model(x.to(device)), 1).cpu()
        conf, idx = probs.max(1)
        ids.extend(batch_ids)
        preds.extend(pp.IDX_TO_CLASS[int(k)] for k in idx)
        confs.extend(conf.tolist())
    return pd.DataFrame({"image_id": ids, "class": preds, "confidence": confs})


def write_submission(pred_df, path):
    """Writes the Kaggle submission and checks it against sample_submission.csv:
    same IDs in the same order, and only the eight official class names."""
    sample = pd.read_csv(pp.SAMPLE_SUB_CSV, dtype={"image_id": str})
    sub = pred_df[["image_id", "class"]]
    assert len(sub) == len(sample), f"Expected {len(sample)} rows, got {len(sub)}"
    assert (sub["image_id"].values == sample["image_id"].values).all(), "IDs or order differ from sample"
    assert set(sub["class"]) <= set(pp.CLASS_NAMES), "Unexpected class names in predictions"
    sub.to_csv(path, index=False)
    print(f"Saved submission: {path} ({len(sub)} rows)")
    return sub