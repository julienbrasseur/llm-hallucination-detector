"""
TCN-based sequence-aware probe for hallucination detection on per-token
LLM activations.

Padding strategy: all padding and dtype conversion happens **on GPU per batch**
during training.  Raw sequences stay as a Python list of CPU float16 tensors
(as loaded from shards).  This avoids expensive CPU tensor allocations on
NUMA machines and keeps memory minimal.
"""

from __future__ import annotations

import copy
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TCNProbeConfig:
    input_dim: int = 4096
    proj_dim: int = 256
    projection: str = "linear"          # "linear" | "pca"

    n_filters: int = 128
    kernel_size: int = 3
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    dropout: float = 0.1
    norm: str = "layer"                 # "layer" | "batch"
    activation: str = "relu"            # "relu" | "gelu"

    pooling: str = "avg"                # "avg" | "max" | "avg+max"
    use_truncation_flag: bool = False
    head_hidden: int = 64
    head_dropout: float = 0.1

    max_seq_len: Optional[int] = 80

    pos_weight: Optional[float] = None
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 64
    patience: int = 5
    scheduler: str = "cosine"           # "cosine" | "plateau" | "none"
    device: str = "auto"
    seed: int = 42

    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)


# ---------------------------------------------------------------------------
# GPU batch padding — the core primitive
# ---------------------------------------------------------------------------
def _pad_batch_gpu(
    sequences: List[torch.Tensor],
    max_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of CPU tensors directly on GPU.

    Returns (padded, mask, truncated) all on ``device``.
    """
    bs = len(sequences)
    hidden = sequences[0].size(1)

    padded = torch.zeros(bs, max_len, hidden, dtype=torch.float16, device=device)
    mask = torch.zeros(bs, max_len, dtype=torch.bool, device=device)
    trunc = torch.zeros(bs, dtype=torch.float32, device=device)

    for j, s in enumerate(sequences):
        L = min(s.size(0), max_len)
        padded[j, :L] = s[:L].to(device)
        mask[j, :L] = True
        if s.size(0) > max_len:
            trunc[j] = 1.0

    return padded, mask, trunc


# ---------------------------------------------------------------------------
# TCN building blocks
# ---------------------------------------------------------------------------
def _get_activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU()}[name]


class _ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout, norm, activation):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.causal_trim = padding

        if norm == "layer":
            self.norm1 = nn.GroupNorm(1, out_ch)
            self.norm2 = nn.GroupNorm(1, out_ch)
        elif norm == "batch":
            self.norm1 = nn.BatchNorm1d(out_ch)
            self.norm2 = nn.BatchNorm1d(out_ch)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.act1 = _get_activation(activation)
        self.act2 = _get_activation(activation)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        if self.causal_trim > 0:
            out = out[:, :, :-self.causal_trim]
        out = self.drop1(self.act1(self.norm1(out)))
        out = self.conv2(out)
        if self.causal_trim > 0:
            out = out[:, :, :-self.causal_trim]
        out = self.drop2(self.act2(self.norm2(out)))
        return out + residual


class _TCNBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        blocks = []
        in_ch = cfg.proj_dim
        for d in cfg.dilations:
            blocks.append(_ResidualBlock(
                in_ch, cfg.n_filters, cfg.kernel_size, d,
                cfg.dropout, cfg.norm, cfg.activation,
            ))
            in_ch = cfg.n_filters
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class _TCNClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.input_dim, cfg.proj_dim) if cfg.projection == "linear" else nn.Identity()
        self.tcn = _TCNBackbone(cfg)

        pool_dim = cfg.n_filters * (2 if cfg.pooling == "avg+max" else 1)
        if cfg.use_truncation_flag:
            pool_dim += 1

        self.head = nn.Sequential(
            nn.Linear(pool_dim, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

    def forward(self, x, mask, truncated=None):
        x = self.proj(x)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)

        mask_f = mask.unsqueeze(1).float()
        denom = mask_f.sum(dim=2, keepdim=True).clamp(min=1.0)

        if self.cfg.pooling in ("avg", "avg+max"):
            avg_pool = (x * mask_f).sum(dim=2) / denom.squeeze(2)
        if self.cfg.pooling in ("max", "avg+max"):
            max_pool = x.masked_fill(~mask.unsqueeze(1), float("-inf")).max(dim=2).values

        if self.cfg.pooling == "avg":
            pooled = avg_pool
        elif self.cfg.pooling == "max":
            pooled = max_pool
        else:
            pooled = torch.cat([avg_pool, max_pool], dim=1)

        if self.cfg.use_truncation_flag and truncated is not None:
            pooled = torch.cat([pooled, truncated.unsqueeze(1)], dim=1)

        return self.head(pooled).squeeze(1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class TCNProbe:
    def __init__(self, **kwargs):
        self.cfg = TCNProbeConfig(**kwargs)
        self._device = self._resolve_device(self.cfg.device)
        self._model: Optional[_TCNClassifier] = None
        self._pca_mean: Optional[torch.Tensor] = None
        self._pca_components: Optional[torch.Tensor] = None
        self._threshold: float = 0.5
        self._fitted: bool = False

    @staticmethod
    def _resolve_device(dev):
        if dev == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(dev)

    # --- PCA -----------------------------------------------------------
    def _fit_pca(self, sequences, verbose=True):
        max_tokens = 50_000
        chunks, count = [], 0
        for s in sequences:
            chunks.append(s.float())
            count += s.size(0)
            if count >= max_tokens:
                break
        mat = torch.cat(chunks, dim=0)[:max_tokens]
        if verbose:
            print(f"PCA: SVD on ({mat.shape[0]}, {mat.shape[1]}) ...")
        mean = mat.mean(dim=0)
        _, _, Vt = torch.linalg.svd(mat - mean, full_matrices=False)
        self._pca_mean = mean
        self._pca_components = Vt[:self.cfg.proj_dim]

    def _apply_pca(self, x):
        return (x.float() - self._pca_mean.to(x.device)) @ self._pca_components.T.to(x.device)

    def _apply_pca_sequences(self, seqs):
        return [self._apply_pca(s) for s in seqs]

    # --- Shard loading -------------------------------------------------
    @staticmethod
    def load_shards(shard_paths):
        all_seqs = []
        for p in sorted(shard_paths):
            data = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(data, dict) and "sequences" in data:
                all_seqs.extend(data["sequences"])
            else:
                raise ValueError(f"Shard {p} not in per-example format.")
        return all_seqs

    # --- Batch iteration helpers ---------------------------------------
    def _iter_batches(self, X, y, bs, shuffle=True):
        """Yield (batch_seqs, batch_labels) from list-of-tensors + labels."""
        n = len(X)
        if shuffle:
            perm = np.random.permutation(n)
        else:
            perm = np.arange(n)

        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            batch_seqs = [X[i] for i in idx]
            batch_y = y[idx] if y is not None else None
            yield batch_seqs, batch_y

    def _pad_and_move(self, batch_seqs):
        """Pad a batch on GPU and return (padded, mask, trunc)."""
        return _pad_batch_gpu(batch_seqs, self.cfg.max_seq_len or 80, self._device)

    # --- Training ------------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        y = np.asarray(y, dtype=np.float32)
        if y_val is not None:
            y_val = np.asarray(y_val, dtype=np.float32)

        # PCA
        if self.cfg.projection == "pca":
            self._fit_pca(X, verbose)
            X = self._apply_pca_sequences(X)
            if X_val is not None:
                X_val = self._apply_pca_sequences(X_val)
            effective_cfg = copy.deepcopy(self.cfg)
            effective_cfg.input_dim = self.cfg.proj_dim
        else:
            effective_cfg = self.cfg

        # Auto pos_weight
        if self.cfg.pos_weight is None:
            n_pos = float(y.sum())
            n_neg = float(len(y) - n_pos)
            pw = n_neg / max(n_pos, 1.0)
            if verbose:
                print(f"Auto pos_weight: {pw:.2f}  (neg={int(n_neg)}, pos={int(n_pos)})")
        else:
            pw = self.cfg.pos_weight

        # Model
        self._model = _TCNClassifier(effective_cfg).to(self._device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=self._device))
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        if self.cfg.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.epochs)
        elif self.cfg.scheduler == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        else:
            sched = None

        use_amp = self._device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        bs = self.cfg.batch_size
        n_train = len(X)
        n_batches_per_epoch = (n_train + bs - 1) // bs

        if verbose:
            n_params = sum(p.numel() for p in self._model.parameters())
            print(f"Model: {n_params:,} params | {n_train:,} train | {n_batches_per_epoch} batches/epoch")

        history = {"train_loss": [], "val_loss": [], "val_auc": []}
        best_val_loss = float("inf")
        best_state = None
        patience_ctr = 0

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            self._model.train()
            epoch_loss, nb = 0.0, 0

            for batch_seqs, batch_y in self._iter_batches(X, y, bs, shuffle=True):
                b_x, b_m, b_t = self._pad_and_move(batch_seqs)
                b_y = torch.tensor(batch_y, dtype=torch.float32, device=self._device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = self._model(b_x, b_m, b_t)
                    loss = criterion(logits, b_y)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                nb += 1

            avg_train = epoch_loss / max(nb, 1)
            history["train_loss"].append(avg_train)

            # Validate
            if X_val is not None and y_val is not None:
                val_loss, val_auc = self._evaluate(X_val, y_val, criterion, use_amp)
                history["val_loss"].append(val_loss)
                history["val_auc"].append(val_auc)
            else:
                val_loss = avg_train
                history["val_loss"].append(None)
                history["val_auc"].append(None)

            if sched is not None:
                sched.step(val_loss) if self.cfg.scheduler == "plateau" else sched.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self._model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1

            if verbose:
                elapsed = time.time() - t0
                msg = f"Epoch {epoch:3d}/{self.cfg.epochs} | train_loss={avg_train:.4f}"
                if history["val_loss"][-1] is not None:
                    msg += f" | val_loss={history['val_loss'][-1]:.4f}"
                if history["val_auc"][-1] is not None:
                    msg += f" | val_AUC={history['val_auc'][-1]:.4f}"
                msg += f" | lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
                if patience_ctr > 0:
                    msg += f" | patience={patience_ctr}/{self.cfg.patience}"
                print(msg)

            if patience_ctr >= self.cfg.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}.")
                break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._fitted = True

        if X_val is not None and y_val is not None:
            self._optimize_threshold(X_val, y_val, use_amp)

        return history

    # --- Evaluation ----------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, X, y, criterion, use_amp):
        self._model.eval()
        bs = self.cfg.batch_size * 2
        total_loss, count = 0.0, 0
        all_probs, all_labels = [], []

        for batch_seqs, batch_y in self._iter_batches(X, y, bs, shuffle=False):
            b_x, b_m, b_t = self._pad_and_move(batch_seqs)
            b_y = torch.tensor(batch_y, dtype=torch.float32, device=self._device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = self._model(b_x, b_m, b_t)
                loss = criterion(logits, b_y)

            total_loss += loss.item() * b_y.size(0)
            count += b_y.size(0)
            all_probs.append(torch.sigmoid(logits.float()).cpu())
            all_labels.append(b_y.cpu())

        probs_np = torch.cat(all_probs).numpy()
        labels_np = torch.cat(all_labels).numpy()

        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels_np, probs_np)
        except Exception:
            auc = 0.0

        return total_loss / max(count, 1), auc

    @torch.no_grad()
    def _optimize_threshold(self, X_val, y_val, use_amp):
        self._model.eval()
        bs = self.cfg.batch_size * 2
        all_probs, all_labels = [], []

        for batch_seqs, batch_y in self._iter_batches(X_val, y_val, bs, shuffle=False):
            b_x, b_m, b_t = self._pad_and_move(batch_seqs)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = self._model(b_x, b_m, b_t)
            all_probs.append(torch.sigmoid(logits.float()).cpu())
            all_labels.append(torch.tensor(batch_y))

        probs = torch.cat(all_probs).numpy()
        labels_np = torch.cat(all_labels).numpy()

        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.1, 0.9, 0.01):
            preds = (probs >= t).astype(float)
            tp = ((preds == 1) & (labels_np == 1)).sum()
            fp = ((preds == 1) & (labels_np == 0)).sum()
            fn = ((preds == 0) & (labels_np == 1)).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self._threshold = float(best_t)

    # --- Inference -----------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, X, batch_size=None):
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        if self.cfg.projection == "pca" and self._pca_components is not None:
            X = self._apply_pca_sequences(X)

        self._model.eval()
        bs = batch_size or self.cfg.batch_size * 2
        use_amp = self._device.type == "cuda"
        all_probs = []

        # No shuffle for inference — use sequential order
        n = len(X)
        for start in range(0, n, bs):
            batch_seqs = X[start:start + bs]
            b_x, b_m, b_t = self._pad_and_move(batch_seqs)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = self._model(b_x, b_m, b_t)
            all_probs.append(torch.sigmoid(logits.float()).cpu())

        return torch.cat(all_probs).numpy()

    def predict(self, X, batch_size=None):
        probs = self.predict_proba(X, batch_size)
        return (probs >= self._threshold).astype(int)

    # --- Serialization -------------------------------------------------
    def save(self, path):
        if not self._fitted:
            warnings.warn("Saving an unfitted probe.")
        torch.save({
            "config": asdict(self.cfg),
            "model_state": self._model.state_dict() if self._model else None,
            "pca_mean": self._pca_mean,
            "pca_components": self._pca_components,
            "threshold": self._threshold,
        }, path)

    @classmethod
    def load(cls, path, device="auto"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg_dict = payload["config"]
        cfg_dict["device"] = device
        probe = cls(**cfg_dict)
        if payload["model_state"] is not None:
            if probe.cfg.projection == "pca":
                eff = copy.deepcopy(probe.cfg)
                eff.input_dim = probe.cfg.proj_dim
            else:
                eff = probe.cfg
            probe._model = _TCNClassifier(eff).to(probe._device)
            probe._model.load_state_dict(payload["model_state"])
            probe._model.eval()
        probe._pca_mean = payload.get("pca_mean")
        probe._pca_components = payload.get("pca_components")
        probe._threshold = payload.get("threshold", 0.5)
        probe._fitted = payload["model_state"] is not None
        return probe

    # --- Diagnostics ---------------------------------------------------
    def summary(self):
        lines = [
            "TCNProbe",
            f"  input_dim       = {self.cfg.input_dim}",
            f"  proj_dim        = {self.cfg.proj_dim} ({self.cfg.projection})",
            f"  n_filters       = {self.cfg.n_filters}",
            f"  kernel_size     = {self.cfg.kernel_size}",
            f"  dilations       = {self.cfg.dilations}",
            f"  receptive_field = {self.cfg.receptive_field()} tokens",
            f"  pooling         = {self.cfg.pooling}",
            f"  max_seq_len     = {self.cfg.max_seq_len}",
            f"  threshold       = {self._threshold:.3f}",
            f"  device          = {self._device}",
            f"  fitted          = {self._fitted}",
        ]
        if self._model is not None:
            n_params = sum(p.numel() for p in self._model.parameters())
            lines.append(f"  total_params    = {n_params:,}")
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()
