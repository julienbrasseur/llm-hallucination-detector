# LLM Hallucination Detector

**llmscan** is a lightweight library for extracting and analyzing LLM internal representations. It was developed for a hallucination detection research project, the idea of which is akin to medical brain imaging: rather than interrogating a model through dialogue (as with a judge LLM), we directly observe its internal activation patterns, much like a PET scan reveals cognitive processes without requiring the subject to speak.

---

**[2026-03-20] New features**

- *Cross-layer last-token extraction:* `extract_last_token` extracts the hidden state of the final non-padding token (typically EOS) at each target layer, returning a tensor of shape `(n_examples, n_layers, hidden_dim)`. Designed for cross-layer probing, where the layer axis becomes the sequence dimension for downstream probes (TCN, attention probe, or flattened for XGBoost). Supports runtime layer override via the `layers` parameter without re-initializing the extractor.
- *Attention probe:* `AttentionProbe` implements a multi-head learned attention mechanism over activation sequences (e.g., cross-layer representations), inspired by [EleutherAI](https://github.com/EleutherAI/attention-probes)'s attention probe architecture. Each head learns which positions (tokens or layers) carry the most signal via a query projection with ALiBi-style position bias, initialized to uniform weighting (equivalent to a mean probe at initialization). Includes `inspect_layer_attention` for visualizing learned attention weights, to better identify which layers the probe deems most informative. (See `attention_probe.py` for details.)

**[2026-03-18] New features**

- *Model-agnostic assistant masking:* All Hugging Face chat models are now supported out of the box (no hardcoded Mistral-specific delimiters). The extractor auto-detects assistant token boundaries from the model's native chat template (via response template matching, with fallback strategies for edge cases). The selected strategy is reported at initialization.
- *Per-example activation shards:* `extract_to_shards` now accepts a `per_example=True` flag, which preserves per-example token boundaries in the output shards (stored as a list of variable-length tensors + a lengths metadata tensor). This is required for sequence-aware probes that operate on raw, non-pooled activations. The default (`per_example=False`) retains the original flat-concatenation format, suitable for training on individual token vectors (e.g., sparse autoencoders).
- *TCN probe:* `TCNProbe` provides a temporal convolutional network for hallucination detection on per-token activation sequences, with an sklearn-style API (`.fit()`, `.predict()`, `.predict_proba()`, `.save()`, `.load()`). Configurable projection, architecture, pooling, and training parameters. (See `tcn.py` for details.)

---

## Getting started

This repository comes with a series of four notebooks illustrating the use of `llmscan` for hallucination detection.
- [`01_Layer_selection.ipynb`](./notebooks/01_Layer_selection.ipynb): Illustrates automatic layer selection with `layer_selection_pipeline`, which extracts mean-pooled activations, trains a probe per layer, and ranks them by F1 and accuracy to identify the most expressive layers.
- [`02_Train_on_a_single_activation_layer.ipynb`](./notebooks/02_Train_on_a_single_activation_layer.ipynb): Illustrates how to train a XGBoost probe on a single activation layer using the `ActivationExtractor` and `XGBoostProbe` classes.
- [`03_Train_on_multiple_activations.ipynb`](./notebooks/03_Train_on_multiple_activations.ipynb): This notebook is analogous to the previous one, but focuses on the case of probe training on multiple activation layers with feature selection.
- [`04_Train_on_attention_layer.ipynb`](./notebooks/04_Train_on_attention_layer.ipynb): Illustrates attention extraction using `AttentionExtractor`, which captures two types of features: multi-head attention outputs and statistical summaries (12 metrics computed per head, including entropy, Gini coefficient, standard deviation, and more experimental measures like Frobenius norm). As in the previous notebooks, these features are used to train XGBoost probes. This final notebook also includes a conclusion summarizing lessons learned across all experiments.

The dataset used throughout these four notebooks is [krogoldAI/hallucination-labeled-dataset](https://huggingface.co/datasets/krogoldAI/hallucination-labeled-dataset). All experiments use [Ministral-8B-Instruct-2410](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410). See the [technical report](./report/technical_report.md) for a detailed summary of findings.

---

## Installation

You can install directly from GitHub:

```bash
pip install git+https://github.com/julienbrasseur/llm-hallucination-detector.git
````

---

## Example usage

All methods taking datasets as input expect OpenAI-type conversation format, i.e., data in the following format:

```json
[
    {
        "conversation": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."}
        ]
    },
    {
        "conversation": [
            {"role": "user", "content": "Show me a Python loop."},
            {"role": "assistant", "content": "Here's a loop:\n```python\nfor i in range(10):\n    print(i)\n```"}
        ]
    }
]
```

### Activation extraction

Activation can be extracted for one or several layers simultaneously using the following:

```py
import os
import torch
from llmscan import ActivationExtractor

# Initialize extractor
extractor = ActivationExtractor(
    model_name="mistralai/Ministral-8B-Instruct-2410",
    target_layers=[16, 17],  # To target simultaneously layers 16 and 17
    device="cuda"
)

# Extract activations
activations = extractor.extract(
    data,  # Dataset to be used for extraction (list of conversation dicts or raw strings)
    batch_size=128,
    max_length=512,
    mean_pool=True,
    focus_on_assistant=True
)

# Save
os.makedirs("feature_cache", exist_ok=True)
torch.save(activations, f"feature_cache/activations_pooled.pt")
```

*Remark:* It is possible to extract raw, non-pooled activations simply by setting `mean_pool=False`. However, please note that this is more memory intensive and requires more available disk space. For large datasets, prefer the `extract_to_shards` method to write activations incrementally:

```py
extractor.extract_to_shards(
    data,  # Dataset to be used for extraction
    out_dir="./shards",
    shard_size_tokens=100_000,
    focus_on_assistant=True,
    per_example=True,
    mean_pool=False
)
```

*Remark:* The `focus_on_assistant` parameter restricts extraction to the last assistant answer exclusively, by masking all preceding tokens. The assistant token boundaries are detected automatically from the model's chat template (see initialization log for the selected masking strategy).

### Attention extraction

Attention can be extracted using a process similar to activation extraction:

```py
import os
import numpy as np
from llmscan import AttentionExtractor

# List of statistics to compute
STATS_TO_COMPUTE = [
    "entropy",
    "std",
    "top5_mass",
    "attention_to_bos"
]

# Initialize attention extractor
extractor = AttentionExtractor(
    model_name="mistralai/Ministral-8B-Instruct-2410",
    target_layers=[16, 17],   # To target simultaneously layers 16 and 17
    stats_to_compute=STATS_TO_COMPUTE,
    extract_mha_output=True,  # To also extract MHA output vectors
    device="cuda",
)

# Extract attention
features = extractor.extract(
    raw_texts=data,  # Dataset to be used for extraction
    batch_size=8,
    max_length=512,
)

# Save attention statistics
os.makedirs("attention_cache", exist_ok=True)
if "attention_stats" in features:
    np.save(f"attention_cache/attention_stats_layer.npy", features["attention_stats"])

# Save MHA outputs
if "mha_output" in features:
    np.save(f"attention_cache/mha_output_layer.npy", features["mha_output"])
```

*Remark:* Attention extractions are mean-pooled by default. Note also that while multi-head attention extraction takes as much time as activation extraction, attention statistics extraction is more compute heavy.

The list of statistics that can be computed can be obtained as follows:

```py
from llmscan import AVAILABLE_STATS

for stat in AVAILABLE_STATS:
    print(stat)
```

Which outputs:

```
entropy
max
std
gini
simpson
effective_token_count
skewness
top1_mass
top5_mass
top10p_mass
mean_relative_distance
attention_to_bos
frobenius
head_output_norm
attn_value_correlation
```

Note that `head_output_norm` and `attn_value_correlation` have not yet been implemented, as they are architecture-dependent. This gap might be filled in a future release.

### Probe training

#### Classification probe (XGBoost)

Training XGBoost probes on activation or attention layers can be done using the `XGBoostProbe` class. The whole process, from training to evaluation, requires train, validation and test splits.

```py
from llmscan import XGBoostProbe

# XGBoost parameters
XGB_PARAMS = {
    "n_estimators": 800,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": "cuda",
    "eval_metric": "logloss",
}

# Train
probe = XGBoostProbe(xgb_params=XGB_PARAMS)
probe.fit(
    x_train, # Train inputs
    y_train, # Train labels
    x_val,   # Validation inputs
    y_val,   # Validation labels
    early_stopping_rounds=20,
    verbose=True
)

# Evaluate on test set
metrics = probe.evaluate(
    x_test,  # Test inputs
    y_test,  # Test labels
    verbose=True
)

# Save probe
probe.save("probe.pkl")
```

Note that inputs must have `numpy` types. If labels are not already in `numpy` format, they will be automatically converted.

#### Sequence-aware probe (TCN)

While mean-pooled probes collapse a variable-length activation sequence into a single vector, some signals may be encoded in the *temporal dynamics* of per-token representations (i.e., how activations evolve across the generated sequence). This is analogous to video classification, where per-frame features must be aggregated into a sequence-level prediction. The temporal order and local patterns carry information that a simple average discards.

`TCNProbe` applies a Temporal Convolutional Network to the full sequence of per-token activations at a given layer. It operates on raw (non-pooled) activations extracted with `per_example=True`, preserving token order and variable sequence lengths.

```py
from llmscan import TCNProbe
import glob

# Load per-example activations (extracted with per_example=True)
X_train = TCNProbe.load_shards(sorted(glob.glob("./shards_train/*.pt")))
X_val = TCNProbe.load_shards(sorted(glob.glob("./shards_val/*.pt")))
X_test = TCNProbe.load_shards(sorted(glob.glob("./shards_test/*.pt")))

# Initialize and train
probe = TCNProbe(
    input_dim=4096,       # hidden dimension of the target model
    proj_dim=256,         # learned projection before convolutions
    n_filters=128,        # channels per conv layer
    kernel_size=3,
    dilations=[1, 2, 4, 8],
    max_seq_len=80,       # truncate sequences beyond this length
    projection="linear",  # or "pca"
    pooling="avg",        # or "max", "avg+max"
)
probe.fit(X_train, train_labels, X_val=X_val, y_val=val_labels)

# Evaluate
probs = probe.predict_proba(X_test)
preds = probe.predict(X_test)

# Save
probe.save("tcn_probe.pt")

# Load
loaded = TCNProbe.load("tcn_probe.pt")
```

All architecture and training parameters are configurable at initialization (see `TCNProbe` docstring for the full list). The probe handles variable-length sequences internally via per-batch GPU padding and mixed-precision training.

#### Sequence-aware probe (Attention Probe)

Where the TCN captures local temporal patterns via convolutions, the `AttentionProbe` takes a complementary approach: it learns *which positions matter most* via multi-head attention. Each head computes a scalar attention weight per position (token or layer), with an ALiBi-style position bias, and aggregates through a learned value projection. The query projection is initialized to zero, so the probe starts as an equivalent of a uniform mean probe and learns to deviate from it during training.

This is particularly useful for cross-layer probing, where the "sequence" is a stack of per-layer representations and the goal is to discover which layers carry the most signal.

```py
from llmscan import (
    AttentionProbe,
    train_attention_probe,
    evaluate_attention_probe,
    inspect_layer_attention,
)

# Cross-layer activations: (n_examples, n_layers, hidden_dim)
# e.g., mean-pooled activations at layers 10–26

model = train_attention_probe(
    X_train, train_labels,
    X_val, val_labels,
    hidden_dim=4096,
    n_heads=8,
    lr=1e-3,
    weight_decay=0.01,
    epochs=100,
    batch_size=256,
    patience=10,
)

# Evaluate on test set
probs, auc, threshold = evaluate_attention_probe(model, X_test, test_labels)

# Get learned layer importance
inspect_layer_attention(model, X_val, layer_indices=list(range(10, 27)))
```

The probe architecture is lightweight (~65K parameters for 8 heads on 4096-dim input) and trains in seconds per epoch. Inspired by [EleutherAI's attention probe](https://github.com/EleutherAI/attention-probes) architecture, reimplemented here as a standalone module with no external dependencies beyond PyTorch.

## License

This project is licensed under the terms of the [MIT license](LICENSE).

## Citation

If you use `llmscan` in your research, please cite:

```bibtex
@software{brasseur2025llmscan,
  author = {Brasseur, Julien},
  title = {llmscan: A Library for LLM Internal Representation Analysis},
  year = {2025},
  url = {https://github.com/julienbrasseur/llm-hallucination-detector}
}
```
