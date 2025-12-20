import os
import json
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

##################################
# ACTIVATION EXTRACTION (cached) #
##################################

class LayerActivationCache:
    """
    Handles extraction + caching of activations for each layer
    to avoid recomputing activations for the validation/test splits.
    """

    def __init__(self, extractor, cache_dir: str = "layer_cache"):
        self.extractor = extractor
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, split: str, layer_idx: int) -> str:
        return os.path.join(self.cache_dir, f"{split}_layer{layer_idx}.pt")

    def get_or_compute(
        self,
        split_name: str,
        data: List[Dict[str, Any]],
        layers: List[int],
        batch_size: int = 8,
        focus_on_assistant: bool = False,
    ) -> Dict[int, torch.Tensor]:
        """
        Loads from cache or computes + saves to cache.
        Returns a dict: { layer_idx: activations[tensor] }
        """

        results = {}

        # Check if all cached
        all_present = all(
            os.path.exists(self._cache_path(split_name, layer))
            for layer in layers
        )

        # If cached, load everything
        if all_present:
            for layer in layers:
                results[layer] = torch.load(self._cache_path(split_name, layer))
            return results

        # Otherwise compute activations ONCE for all layers (efficient)
        print(f"[Extracting {split_name}] Computing activations for layers: {layers}")

        # Extract concatenated activations for all layers first:
        acts = self.extractor.extract(
            raw_texts=data,
            batch_size=batch_size,
            focus_on_assistant=focus_on_assistant,
            mean_pool=True
        )  # shape: [N, len(layers) * dim]

        # Now slice per-layer
        hidden_dim = acts.size(1) // len(layers)
        for i, layer in enumerate(layers):
            layer_acts = acts[:, i * hidden_dim:(i + 1) * hidden_dim].clone()
            torch.save(layer_acts, self._cache_path(split_name, layer))
            results[layer] = layer_acts

        return results


#######################################
# TRAIN/EVAL FOR EACH LAYER (XGBoost) #
#######################################

class LayerClassifierEvaluator:
    """
    Trains + evaluates one small XGBoost classifier per layer.
    """

    def __init__(self, labels_val, labels_test):
        self.y_val = np.array(labels_val)
        self.y_test = np.array(labels_test)

    def train_single_layer(self, X_val, X_test) -> Dict[str, float]:
        """
        Train a single small XGBoost model on X_val then evaluate on X_test.
        """
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda" if torch.cuda.is_available() else "cpu",
            eval_metric="logloss",
        )

        model.fit(
            X_val, self.y_val,
            eval_set=[(X_val, self.y_val)],
            verbose=False
        )

        preds = model.predict(X_test)

        return {
            "acc": accuracy_score(self.y_test, preds),
            "f1": f1_score(self.y_test, preds, average="weighted"),
        }


#################
# FULL PIPELINE #
#################

def layer_selection_pipeline(
    extractor,
    layers: List[int],
    val_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    labels_val: List[int],
    labels_test: List[int],
    batch_size: int = 8,
    focus_on_assistant: bool = False,
    plot_results: bool = True,
    save_path: Optional[str] = None,
):
    """
    Full pipeline:
    - Extract & cache validation/test activations for each layer
    - Train one classifier per layer
    - Evaluate on test set
    - Plot results
    - Return ranking
    """

    cache = LayerActivationCache(extractor)

    # Compute activations
    val_acts = cache.get_or_compute(
        "val", val_data, layers,
        batch_size=batch_size,
        focus_on_assistant=focus_on_assistant,
    )
    test_acts = cache.get_or_compute(
        "test", test_data, layers,
        batch_size=batch_size,
        focus_on_assistant=focus_on_assistant,
    )

    evaluator = LayerClassifierEvaluator(labels_val, labels_test)

    results = {}

    # Train and evaluate
    for layer in tqdm(layers, desc="Evaluating layers"):
        Xv = val_acts[layer].float().numpy()
        Xt = test_acts[layer].float().numpy()

        metrics = evaluator.train_single_layer(Xv, Xt)
        results[layer] = metrics

    # Ranking
    ranked = sorted(results.items(), key=lambda x: (-x[1]["f1"], -x[1]["acc"]))

    print("\n===== Layer Ranking (Best → Worst) =====")
    for layer, m in ranked:
        print(f"Layer {layer}: F1={m['f1']:.4f} | Acc={m['acc']:.4f}")

    # Visualization
    if plot_results:
        _plot_layer_performance(results, save_path)

    return ranked, results


def _plot_layer_performance(metrics: Dict[int, Dict[str, float]], save_path: Optional[str] = None):
    """Plot F1 and Accuracy across layers."""
    layers = sorted(metrics.keys())
    f1_scores = [metrics[l]['f1'] for l in layers]
    accuracy = [metrics[l]['acc'] for l in layers]

    plt.figure(figsize=(10, 6))
    plt.plot(layers, f1_scores, marker='o', label='F1 Score')
    plt.plot(layers, accuracy, marker='s', label='Accuracy')

    plt.xlabel("Layer")
    plt.ylabel("Metric Value")
    plt.title("Layer-wise F1 Score and Accuracy")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.xticks(layers)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()