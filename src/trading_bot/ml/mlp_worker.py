"""Isolated worker for the PyTorch MLP v2 baseline.

Runs in its own process because LightGBM (Homebrew libomp) and torch
(bundled libomp) deadlock when they share one process on macOS — the
duplicate-OpenMP-runtime problem, and KMP_DUPLICATE_LIB_OK does not
reliably clear it. This module must never import lightgbm.

Protocol: `python -m trading_bot.ml.mlp_worker <in.pkl> <out.pkl>`
where in.pkl is a pickled dict {df, folds: [(train_dates, val_dates)],
midpoints, seed} and out.pkl receives the pooled OOS frame with p_*
probability columns, score and pred_class_idx.
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd


def run(payload: dict) -> pd.DataFrame:
    import torch
    from torch import nn

    from trading_bot.ml import features as F
    from trading_bot.ml.train import CLASSES, N_CLASSES, score_from_probs

    df: pd.DataFrame = payload["df"]
    folds = payload["folds"]
    midpoints = payload["midpoints"]
    seed = payload["seed"]

    torch.manual_seed(seed)
    torch.set_num_threads(4)
    Xcols = F.FEATURE_COLUMNS
    parts = []
    for k, (train_dates, val_dates) in enumerate(folds):
        tr = df[df["date"].isin(set(train_dates))]
        va = df[df["date"].isin(set(val_dates))]
        mu = tr[Xcols].mean().to_numpy()
        sd = tr[Xcols].std().replace(0, 1).to_numpy()
        Xtr = torch.tensor((tr[Xcols].to_numpy() - mu) / sd, dtype=torch.float32)
        ytr = torch.tensor(tr["y"].to_numpy(), dtype=torch.long)
        Xva = torch.tensor((va[Xcols].to_numpy() - mu) / sd, dtype=torch.float32)
        yva = torch.tensor(va["y"].to_numpy(), dtype=torch.long)

        counts = np.bincount(tr["y"].to_numpy(), minlength=N_CLASSES).astype(float)
        counts[counts == 0] = 1.0
        weight = torch.tensor(len(tr) / (N_CLASSES * counts), dtype=torch.float32)

        model = nn.Sequential(
            nn.Linear(len(Xcols), 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, N_CLASSES),
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss(weight=weight)

        best_val = float("inf")
        best_state = None
        patience = 3
        stale = 0
        for _epoch in range(30):
            model.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), 4096):
                idx = perm[i : i + 4096]
                opt.zero_grad()
                loss = loss_fn(model(Xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(Xva), yva))
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(Xva), dim=1).numpy()

        part = va[["date", "ticker", "actual_return_pct", "y"]].copy()
        for i, cls in enumerate(CLASSES):
            part[f"p_{cls}"] = probs[:, i]
        part["score"] = score_from_probs(probs, midpoints)
        part["pred_class_idx"] = probs.argmax(axis=1)
        parts.append(part)
        print(f"mlp worker: fold {k + 1}/{len(folds)} done (val loss {best_val:.4f})",
              file=sys.stderr)
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path, "rb") as f:
        payload = pickle.load(f)
    oos = run(payload)
    with open(out_path, "wb") as f:
        pickle.dump(oos, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
