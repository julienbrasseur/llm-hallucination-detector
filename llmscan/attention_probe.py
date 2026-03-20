"""
Attention probe for cross-layer hallucination detection.

Reimplements the EleutherAI attention probe architecture:
- Multi-head learned attention over the layer axis
- ALiBi-style position bias (learned per head)
- query_proj initialized to zero → starts as uniform mean probe
- Value projection → weighted sum → binary classification

Usage:
    Expects cross-layer activations as (N, n_layers, hidden_dim) tensors
    saved in layer_cache/ and labels from the HF dataset.
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, classification_report


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AttentionProbe(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_classes=1):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, n_heads, bias=False)
        self.value_proj = nn.Linear(hidden_dim, n_heads * n_classes, bias=False)
        self.position_weights = nn.Parameter(torch.zeros(n_heads))
        self.n_heads = n_heads
        self.n_classes = n_classes

        # Init query to zero → starts as uniform mean probe
        nn.init.zeros_(self.query_proj.weight)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, hidden_dim)
        seq_len = x.size(1)

        # Attention logits: (batch, seq_len, n_heads)
        logits = self.query_proj(x)

        # ALiBi-style position bias (relative to first position)
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        logits = logits + self.position_weights[None, None, :] * positions[:, None]

        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))

        weights = torch.softmax(logits, dim=1)  # (batch, seq_len, n_heads)

        # Values: (batch, seq_len, n_heads, n_classes)
        values = self.value_proj(x).view(
            x.size(0), seq_len, self.n_heads, self.n_classes
        )

        # Weighted sum over seq and heads: (batch, n_classes)
        output = (weights.unsqueeze(-1) * values).sum(dim=1).sum(dim=1)
        return output.squeeze(-1)

    def get_attention_weights(self, x):
        """Return attention weights for inspection (which layers matter)."""
        seq_len = x.size(1)
        logits = self.query_proj(x)
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        logits = logits + self.position_weights[None, None, :] * positions[:, None]
        return torch.softmax(logits, dim=1)  # (batch, seq_len, n_heads)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_attention_probe(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_dim=4096,
    n_heads=8,
    lr=1e-3,
    weight_decay=0.01,
    epochs=100,
    batch_size=256,
    patience=10,
    device="cuda",
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Tensors
    X_train_t = X_train.float().to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = X_val.float().to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    # Class weighting
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    print(f"Auto pos_weight: {pos_weight.item():.2f}  (neg={int(n_neg)}, pos={int(n_pos)})")

    # Model
    model = AttentionProbe(hidden_dim, n_heads, n_classes=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AttentionProbe: {n_params:,} params | {n_heads} heads | hidden_dim={hidden_dim}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_train = len(X_train_t)
    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        # Shuffle
        perm = torch.randperm(n_train, device=device)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            bx = X_train_t[idx]
            by = y_train_t[idx]

            logits = model(bx)
            loss = criterion(logits, by)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1)

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = roc_auc_score(y_val, val_probs)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        elapsed = time.time() - t0
        msg = (
            f"Epoch {epoch:3d}/{epochs} | train_loss={avg_train:.4f} | "
            f"val_loss={val_loss:.4f} | val_AUC={val_auc:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
        )
        if patience_ctr > 0:
            msg += f" | patience={patience_ctr}/{patience}"
        print(msg)

        if patience_ctr >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_attention_probe(model, X_test, y_test, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()

    X_test_t = X_test.float().to(device)
    logits = model(X_test_t)
    probs = torch.sigmoid(logits).cpu().numpy()

    auc = roc_auc_score(y_test, probs)
    print(f"\nAUC: {auc:.4f}")

    # Optimize threshold
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= t).astype(float)
        tp = ((preds == 1) & (y_test == 1)).sum()
        fp = ((preds == 1) & (y_test == 0)).sum()
        fn = ((preds == 0) & (y_test == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"Optimized threshold: {best_t:.3f}")

    preds_default = (probs >= 0.5).astype(int)
    preds_optimized = (probs >= best_t).astype(int)

    print("\n--- Default threshold (0.5) ---")
    print(classification_report(y_test, preds_default, target_names=["correct", "hallucinated"]))

    print("--- Optimized threshold ---")
    print(classification_report(y_test, preds_optimized, target_names=["correct", "hallucinated"]))

    return probs, auc, best_t


# ---------------------------------------------------------------------------
# Inspect learned attention weights
# ---------------------------------------------------------------------------
@torch.no_grad()
def inspect_layer_attention(model, X, layer_indices, device="cuda"):
    """Print which layers the probe learned to attend to."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()
    X_t = X.float().to(device)

    # Average attention weights across all examples
    weights = model.get_attention_weights(X_t)  # (N, n_layers, n_heads)
    avg_weights = weights.mean(dim=0).cpu().numpy()  # (n_layers, n_heads)

    print("\n--- Learned layer attention weights (averaged over examples) ---")
    print(f"{'Layer':>8}", end="")
    for h in range(avg_weights.shape[1]):
        print(f"  Head {h:d}", end="")
    print(f"  {'Mean':>6}")
    print("-" * (10 + 8 * avg_weights.shape[1] + 8))

    for i, l in enumerate(layer_indices):
        print(f"Layer {l:>2d}", end="")
        for h in range(avg_weights.shape[1]):
            print(f"  {avg_weights[i, h]:.4f}", end="")
        print(f"  {avg_weights[i].mean():.4f}")