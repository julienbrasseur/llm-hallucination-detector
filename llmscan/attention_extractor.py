"""
Comprehensive Attention Feature Extractor for Hallucination Detection.

Extracts interpretable features from transformer attention patterns and MHA outputs,
including entropy, Gini, concentration metrics, positional statistics, and head-level norms.

Usage:
    extractor = AttentionExtractor(
        model_name="mistralai/Ministral-8B-Instruct-2410",
        target_layers=[16],
        stats_to_compute=["entropy", "max", "gini", "top5_mass", "effective_token_count"],
        extract_mha_output=True,
        device='cuda',
    )
    
    features = extractor.extract(dataset_texts)
    # Returns: dict with 'attention_stats' and/or 'mha_output' arrays
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from scipy.stats import skew as scipy_skew
import warnings


# All available statistics
AVAILABLE_STATS = [
    # Row-level stats (per head, averaged over assistant tokens)
    "entropy",               # Attention diffuseness
    "max",                    # Strongest attention weight
    "std",                    # Standard deviation of attention
    "gini",                   # True Gini coefficient (sorting-based)
    "simpson",                # Sum of squared attention (p²)
    "effective_token_count",  # 1/simpson - interpretable sparsity
    "skewness",               # Asymmetry of attention distribution
    "top1_mass",              # Largest attention weight
    "top5_mass",              # Sum of top 5 attention weights
    "top10p_mass",            # Sum of top 10% attention weights
    "mean_relative_distance", # How far back attention looks
    "attention_to_bos",       # Attention to position 0 (BOS token)
    
    # Matrix-level stats (per head, on assistant submatrix)
    "frobenius",              # Matrix energy across generation
    
    # Head output stats (require deeper hooks)
    "head_output_norm",       # L2 norm of head output
    "attn_value_correlation", # Correlation between attention and value norms
]

# Stats that require only attention weights
ATTENTION_WEIGHT_STATS = [
    "entropy", "max", "std", "gini", "simpson", "effective_token_count",
    "skewness", "top1_mass", "top5_mass", "top10p_mass",
    "mean_relative_distance", "attention_to_bos", "frobenius",
]

# Stats that require head outputs / value vectors
HEAD_OUTPUT_STATS = ["head_output_norm", "attn_value_correlation"]


class AttentionExtractor:
    """
    Unified extractor for attention-based features and MHA outputs.
    
    Supports:
    - Row-level attention statistics (entropy, Gini, top-k, etc.)
    - Matrix-level statistics (Frobenius norm)
    - Head output statistics (norms, correlations)
    - MHA output vectors (mean-pooled over assistant tokens)
    """
    
    def __init__(
        self,
        model_name: str,
        target_layers: List[int],
        stats_to_compute: Optional[List[str]] = None,
        extract_mha_output: bool = False,
        device: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the attention extractor.
        
        Args:
            model_name: HuggingFace model identifier
            target_layers: List of layer indices to extract from
            stats_to_compute: List of statistics to compute. If None, computes all.
                            See AVAILABLE_STATS for options.
            extract_mha_output: Whether to extract mean-pooled MHA output vectors
            device: Device to use ('cuda', 'cpu', or specific 'cuda:0')
            tokenizer_kwargs: Additional kwargs for tokenizer
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_layers = target_layers
        self.extract_mha_output = extract_mha_output
        
        # Validate and set stats to compute
        if stats_to_compute is None:
            self.stats_to_compute = AVAILABLE_STATS.copy()
        else:
            invalid = set(stats_to_compute) - set(AVAILABLE_STATS)
            if invalid:
                raise ValueError(f"Unknown stats: {invalid}. Available: {AVAILABLE_STATS}")
            self.stats_to_compute = stats_to_compute
        
        # Check if we need head outputs
        self.need_head_outputs = any(s in HEAD_OUTPUT_STATS for s in self.stats_to_compute)
        self.need_attention_weights = any(s in ATTENTION_WEIGHT_STATS for s in self.stats_to_compute)
        
        # Storage for hooked values
        self._hooked_head_outputs = {}
        self._hooked_value_vectors = {}
        self._hooked_mha_outputs = {}
        self._hooks = []
        
        print(f"Loading model...")
        print(f"  Stats to compute: {self.stats_to_compute}")
        print(f"  Extract MHA output: {extract_mha_output}")
        print(f"  Need head outputs: {self.need_head_outputs}")
        
        # Load model with appropriate outputs enabled
        self.model = AutoModel.from_pretrained(
            model_name,
            output_attentions=self.need_attention_weights,
            output_hidden_states=False,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            **(tokenizer_kwargs or {})
        )
        
        # Setup padding
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.padding_side = "right"
        
        # Find instruction end token for assistant mask
        self._setup_instruction_token()
        
        # Register hooks if needed
        if self.need_head_outputs or self.extract_mha_output:
            self._register_hooks()
        
        # Get model config
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.num_heads
        self.hidden_size = self.model.config.hidden_size
        
        print(f"Model loaded: {self.num_heads} heads, dim={self.hidden_size}")
        print(f"Target layers: {target_layers}")
    
    def _setup_instruction_token(self):
        """Identify the instruction-end token for creating assistant masks."""
        self.end_token_id = None
        
        # Try common instruction end tokens
        candidates = ["[/INST]", "</s>", "<|assistant|>", "<|im_end|>"] # to do: simplify this
        for token in candidates:
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id != self.tokenizer.unk_token_id:
                    self.end_token_id = token_id
                    print(f"Using instruction end token: {token}")
                    break
            except Exception:
                continue
    
    def _register_hooks(self):
        """Register forward hooks to capture head outputs and MHA outputs."""
        
        def make_head_output_hook(layer_idx):
            def hook(module, input, output):
                # For most transformer implementations, we need to access
                # the attention output before the final projection
                # This is model-specific
                pass
            return hook
        
        def make_mha_output_hook(layer_idx):
            def hook(module, input, output):
                # Capture the full MHA output (after projection)
                # output is typically (hidden_states, attention_weights, ...)
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                self._hooked_mha_outputs[layer_idx] = hidden.detach()
            return hook
        
        # Register hooks on attention layers
        for layer_idx in self.target_layers:
            layer = self.model.layers[layer_idx]
            
            if self.extract_mha_output:
                # Hook the self_attn module to get MHA output
                hook = layer.self_attn.register_forward_hook(make_mha_output_hook(layer_idx))
                self._hooks.append(hook)
            
            if self.need_head_outputs:
                # For head-level outputs, we need model-specific handling
                # This will be implemented in _extract_head_outputs
                pass
    
    def _clear_hooks(self):
        """Clear stored hooked values."""
        self._hooked_head_outputs.clear()
        self._hooked_value_vectors.clear()
        self._hooked_mha_outputs.clear()
    
    def _create_assistant_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Create binary mask where 1 = assistant token, 0 = other."""
        batch_size, seq_len = input_ids.shape
        assistant_mask = torch.zeros_like(attention_mask)
        
        if self.end_token_id is not None:
            is_end = (input_ids == self.end_token_id)
            for b in range(batch_size):
                end_positions = is_end[b].nonzero(as_tuple=True)[0]
                if len(end_positions) > 0:
                    # Start after the last instruction end token
                    start = end_positions[-1].item() + 1
                    seq_length = int(attention_mask[b].sum().item())
                    if start < seq_length:
                        assistant_mask[b, start:seq_length] = 1
                else:
                    # No end token found - assume all tokens are assistant
                    seq_length = int(attention_mask[b].sum().item())
                    assistant_mask[b, :seq_length] = 1
        else:
            # Fallback: assume second half is assistant
            for b in range(batch_size):
                seq_length = int(attention_mask[b].sum().item())
                mid = seq_length // 2
                assistant_mask[b, mid:seq_length] = 1
        
        return assistant_mask
    
    def _format_conversation(self, item: Union[Dict[str, Any], str]) -> str:
        """Convert item to conversation string."""
        if isinstance(item, str):
            return item
        
        if isinstance(item, dict) and "conversation" in item:
            conv = item["conversation"]
            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    return self.tokenizer.apply_chat_template(
                        conv, tokenize=False, add_generation_prompt=False
                    )
                except Exception:
                    pass
            
            # Fallback formatting
            parts = []
            for turn in conv:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                parts.append(f"<{role}>: {content}")
            return "\n".join(parts)
        
        raise ValueError(f"Unsupported item type: {type(item)}")
    
    ##########################
    # STATISTICS COMPUTATION #
    ##########################
    
    def _compute_entropy(self, p: torch.Tensor) -> torch.Tensor:
        """Compute Shannon entropy: -sum(p * log(p))."""
        p = p + 1e-12  # Avoid log(0)
        return -(p * torch.log(p)).sum(dim=-1)
    
    def _compute_gini(self, p: torch.Tensor) -> torch.Tensor:
        """
        Compute Gini coefficient (sorting-based).
        
        Gini = (2 * sum(i * p_sorted[i])) / (n * sum(p)) - (n + 1) / n
        
        Args:
            p: Attention distribution [..., seq_len]
        
        Returns:
            Gini coefficient [...] 
        """
        # Sort along last dimension
        p_sorted, _ = torch.sort(p, dim=-1)
        n = p.shape[-1]
        
        # Create index weights [1, 2, 3, ..., n]
        indices = torch.arange(1, n + 1, device=p.device, dtype=p.dtype)
        
        # Compute Gini
        # Sum of (index * sorted_value) along last dim
        weighted_sum = (indices * p_sorted).sum(dim=-1)
        total_sum = p_sorted.sum(dim=-1) + 1e-12
        
        gini = (2 * weighted_sum) / (n * total_sum) - (n + 1) / n
        return gini
    
    def _compute_simpson(self, p: torch.Tensor) -> torch.Tensor:
        """Compute Simpson index: sum(p²)."""
        return (p ** 2).sum(dim=-1)
    
    def _compute_effective_token_count(self, p: torch.Tensor) -> torch.Tensor:
        """Compute effective token count: 1 / sum(p²)."""
        simpson = self._compute_simpson(p)
        return 1.0 / (simpson + 1e-12)
    
    def _compute_skewness(self, p: torch.Tensor) -> torch.Tensor:
        """Compute skewness (third standardized moment)."""
        # p: [..., seq_len]
        mean = p.mean(dim=-1, keepdim=True)
        std = p.std(dim=-1, keepdim=True) + 1e-12
        skew = ((p - mean) ** 3).mean(dim=-1) / (std.squeeze(-1) ** 3)
        return skew
    
    def _compute_top_k_mass(self, p: torch.Tensor, k: int) -> torch.Tensor:
        """Compute sum of top-k attention weights."""
        topk_values, _ = torch.topk(p, min(k, p.shape[-1]), dim=-1)
        return topk_values.sum(dim=-1)
    
    def _compute_top_percent_mass(self, p: torch.Tensor, percent: float) -> torch.Tensor:
        """Compute sum of top X% attention weights."""
        k = max(1, int(p.shape[-1] * percent))
        return self._compute_top_k_mass(p, k)
    
    def _compute_mean_relative_distance(
        self,
        p: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mean relative distance weighted by attention.
        
        Args:
            p: Attention weights [num_heads, num_query, seq_len]
            query_positions: Positions of query tokens [num_query]
        
        Returns:
            Mean relative distance [num_heads, num_query]
        """
        num_heads, num_query, seq_len = p.shape
        
        # Key positions: [seq_len]
        key_positions = torch.arange(seq_len, device=p.device, dtype=p.dtype)
        
        # Relative distance: query_pos - key_pos (positive = looking back)
        # query_positions: [num_query] -> [1, num_query, 1]
        # key_positions: [seq_len] -> [1, 1, seq_len]
        query_pos = query_positions.view(1, -1, 1).float()
        key_pos = key_positions.view(1, 1, -1)
        
        relative_dist = query_pos - key_pos  # [1, num_query, seq_len]
        
        # Weighted mean: sum(p * dist) / sum(p)
        weighted_dist = (p * relative_dist).sum(dim=-1)  # [num_heads, num_query]
        
        return weighted_dist
    
    def _compute_frobenius_norm(self, A: torch.Tensor) -> torch.Tensor:
        """
        Compute Frobenius norm of attention submatrix.
        
        Args:
            A: Attention submatrix [num_heads, num_assistant, seq_len]
        
        Returns:
            Frobenius norm per head [num_heads]
        """
        return torch.sqrt((A ** 2).sum(dim=(-2, -1)))
    
    def _compute_row_level_stats(
        self,
        attention: torch.Tensor,
        assistant_indices: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all row-level statistics.
        
        Args:
            attention: [num_heads, seq_len, seq_len]
            assistant_indices: Indices of assistant tokens
            padding_mask: [seq_len] valid token mask
        
        Returns:
            Dict mapping stat name to [num_heads] tensor (averaged over assistant tokens)
        """
        num_heads, seq_len, _ = attention.shape
        
        # Extract attention rows for assistant tokens as queries
        # Shape: [num_heads, num_assistant, seq_len]
        assistant_attn = attention[:, assistant_indices, :]
        num_assistant = len(assistant_indices)
        
        if num_assistant == 0:
            # Return zeros if no assistant tokens
            return {stat: torch.zeros(num_heads, device=attention.device) 
                    for stat in self.stats_to_compute if stat in ATTENTION_WEIGHT_STATS}
        
        stats = {}
        
        # Entropy
        if "entropy" in self.stats_to_compute:
            entropy = self._compute_entropy(assistant_attn)  # [heads, num_asst]
            stats["entropy"] = entropy.mean(dim=1)  # [heads]
        
        # Max
        if "max" in self.stats_to_compute:
            max_attn = assistant_attn.max(dim=-1)[0]  # [heads, num_asst]
            stats["max"] = max_attn.mean(dim=1)
        
        # Top1 mass (same as max but explicit)
        if "top1_mass" in self.stats_to_compute:
            stats["top1_mass"] = assistant_attn.max(dim=-1)[0].mean(dim=1)
        
        # Std
        if "std" in self.stats_to_compute:
            std_attn = assistant_attn.std(dim=-1)  # [heads, num_asst]
            stats["std"] = std_attn.mean(dim=1)
        
        # Gini
        if "gini" in self.stats_to_compute:
            gini = self._compute_gini(assistant_attn)  # [heads, num_asst]
            stats["gini"] = gini.mean(dim=1)
        
        # Simpson
        if "simpson" in self.stats_to_compute:
            simpson = self._compute_simpson(assistant_attn)
            stats["simpson"] = simpson.mean(dim=1)
        
        # Effective token count
        if "effective_token_count" in self.stats_to_compute:
            etc = self._compute_effective_token_count(assistant_attn)
            stats["effective_token_count"] = etc.mean(dim=1)
        
        # Skewness
        if "skewness" in self.stats_to_compute:
            skew = self._compute_skewness(assistant_attn)
            stats["skewness"] = skew.mean(dim=1)
        
        # Top-5 mass
        if "top5_mass" in self.stats_to_compute:
            top5 = self._compute_top_k_mass(assistant_attn, 5)
            stats["top5_mass"] = top5.mean(dim=1)
        
        # Top-10% mass
        if "top10p_mass" in self.stats_to_compute:
            top10p = self._compute_top_percent_mass(assistant_attn, 0.10)
            stats["top10p_mass"] = top10p.mean(dim=1)
        
        # Mean relative distance
        if "mean_relative_distance" in self.stats_to_compute:
            mrd = self._compute_mean_relative_distance(assistant_attn, assistant_indices)
            stats["mean_relative_distance"] = mrd.mean(dim=1)
        
        # Attention to BOS
        if "attention_to_bos" in self.stats_to_compute:
            attn_bos = assistant_attn[:, :, 0]  # [heads, num_asst]
            stats["attention_to_bos"] = attn_bos.mean(dim=1)
        
        return stats
    
    def _compute_matrix_level_stats(
        self,
        attention: torch.Tensor,
        assistant_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute matrix-level statistics.
        
        Args:
            attention: [num_heads, seq_len, seq_len]
            assistant_indices: Indices of assistant tokens
        
        Returns:
            Dict mapping stat name to [num_heads] tensor
        """
        stats = {}
        
        if "frobenius" in self.stats_to_compute:
            if len(assistant_indices) > 0:
                # Extract [heads, num_assistant, seq_len] submatrix
                assistant_attn = attention[:, assistant_indices, :]
                frob = self._compute_frobenius_norm(assistant_attn)
            else:
                frob = torch.zeros(attention.shape[0], device=attention.device)
            stats["frobenius"] = frob
        
        return stats
    
    def _compute_head_output_stats(
        self,
        layer_idx: int,
        assistant_mask: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute head-level output statistics.
        
        Note: This requires model-specific handling and hooked values.
        
        Args:
            layer_idx: Layer index
            assistant_mask: [seq_len] mask for assistant tokens
            attention_weights: [num_heads, seq_len, seq_len] if needed for correlation
        
        Returns:
            Dict mapping stat name to [num_heads] tensor
        """
        stats = {}
        
        # These stats require access to per-head outputs which need custom hooks
        # For now, we'll compute approximations or skip if not available
        
        if "head_output_norm" in self.stats_to_compute:
            # This would require hooking into the attention output before projection
            # and computing per-head norms
            # Placeholder - will be implemented with proper hooks
            warnings.warn("head_output_norm requires custom hooks - returning zeros")
            stats["head_output_norm"] = torch.zeros(self.num_heads)
        
        if "attn_value_correlation" in self.stats_to_compute:
            # This requires access to value vectors
            # Placeholder
            warnings.warn("attn_value_correlation requires custom hooks - returning zeros")
            stats["attn_value_correlation"] = torch.zeros(self.num_heads)
        
        return stats
    
    def _extract_features_single_sample(
        self,
        attention: torch.Tensor,
        assistant_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        layer_idx: int,
    ) -> np.ndarray:
        """
        Extract all attention statistics for a single sample.
        
        Args:
            attention: [num_heads, seq_len, seq_len]
            assistant_mask: [seq_len]
            padding_mask: [seq_len]
            layer_idx: Current layer index
        
        Returns:
            Feature vector as numpy array
        """
        # Get assistant token indices
        assistant_indices = assistant_mask.bool().nonzero(as_tuple=True)[0]
        
        all_stats = {}
        
        # Row-level stats
        if self.need_attention_weights:
            row_stats = self._compute_row_level_stats(
                attention, assistant_indices, padding_mask
            )
            all_stats.update(row_stats)
            
            # Matrix-level stats
            matrix_stats = self._compute_matrix_level_stats(attention, assistant_indices)
            all_stats.update(matrix_stats)
        
        # Head output stats
        if self.need_head_outputs:
            head_stats = self._compute_head_output_stats(
                layer_idx, assistant_mask, attention
            )
            all_stats.update(head_stats)
        
        # Flatten to feature vector: [stat1_head0, stat1_head1, ..., stat2_head0, ...]
        features = []
        for stat_name in sorted(all_stats.keys()):
            stat_values = all_stats[stat_name]
            features.extend(stat_values.float().cpu().numpy().tolist())
        
        return np.array(features, dtype=np.float32)
    
    def _extract_mha_output_single_sample(
        self,
        mha_output: torch.Tensor,
        assistant_mask: torch.Tensor,
    ) -> np.ndarray:
        """
        Extract mean-pooled MHA output for assistant tokens.
        
        Args:
            mha_output: [seq_len, hidden_dim]
            assistant_mask: [seq_len]
        
        Returns:
            Mean-pooled vector [hidden_dim]
        """
        assistant_indices = assistant_mask.bool().nonzero(as_tuple=True)[0]
        
        if len(assistant_indices) == 0:
            return np.zeros(self.hidden_size, dtype=np.float32)
        
        # Extract and mean-pool
        assistant_outputs = mha_output[assistant_indices, :]  # [num_asst, hidden]
        pooled = assistant_outputs.mean(dim=0)  # [hidden]
        
        return pooled.float().cpu().numpy()
    
    def extract(
        self,
        raw_texts: List[Union[Dict[str, Any], str]],
        batch_size: int = 8,
        max_length: int = 1024,
    ) -> Dict[str, np.ndarray]:
        """
        Extract attention features and/or MHA outputs for all samples.
        
        Args:
            raw_texts: List of conversation dicts or strings
            batch_size: Batch size for processing
            max_length: Maximum sequence length
        
        Returns:
            Dict with keys:
                - 'attention_stats': [N, num_features] if stats requested
                - 'mha_output': [N, hidden_dim * num_layers] if extract_mha_output=True
        """
        texts = [self._format_conversation(x) for x in raw_texts]
        
        all_attention_stats = []
        all_mha_outputs = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting attention features"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            
            # Clear hooked values
            self._clear_hooks()
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Create assistant masks
            assistant_masks = self._create_assistant_mask(
                inputs["input_ids"],
                inputs["attention_mask"],
            )
            
            # Process each sample in batch
            for b in range(len(batch_texts)):
                sample_features_per_layer = []
                sample_mha_per_layer = []
                
                for layer_idx in self.target_layers:
                    # Attention statistics
                    if self.need_attention_weights:
                        layer_attn = outputs.attentions[layer_idx][b]  # [heads, seq, seq]
                        layer_attn = layer_attn.float().cpu()
                        
                        features = self._extract_features_single_sample(
                            layer_attn,
                            assistant_masks[b].cpu(),
                            inputs["attention_mask"][b].cpu(),
                            layer_idx,
                        )
                        sample_features_per_layer.append(features)
                    
                    # MHA output
                    if self.extract_mha_output and layer_idx in self._hooked_mha_outputs:
                        mha_out = self._hooked_mha_outputs[layer_idx][b]  # [seq, hidden]
                        mha_out = mha_out.float().cpu()
                        
                        pooled_mha = self._extract_mha_output_single_sample(
                            mha_out,
                            assistant_masks[b].cpu(),
                        )
                        sample_mha_per_layer.append(pooled_mha)
                
                # Concatenate across layers
                if sample_features_per_layer:
                    all_attention_stats.append(np.concatenate(sample_features_per_layer))
                
                if sample_mha_per_layer:
                    all_mha_outputs.append(np.concatenate(sample_mha_per_layer))
            
            # Free memory
            del outputs
            torch.cuda.empty_cache()
        
        # Build result dict
        result = {}
        
        if all_attention_stats:
            result['attention_stats'] = np.vstack(all_attention_stats)
            print(f"Attention stats shape: {result['attention_stats'].shape}")
        
        if all_mha_outputs:
            result['mha_output'] = np.vstack(all_mha_outputs)
            print(f"MHA output shape: {result['mha_output'].shape}")
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """
        Get human-readable names for all features.
        
        Returns:
            List of feature names in order they appear in extracted features.
        """
        names = []
        
        # Stats are sorted alphabetically, each has num_heads values
        stats_in_order = sorted([s for s in self.stats_to_compute 
                                  if s in ATTENTION_WEIGHT_STATS or s in HEAD_OUTPUT_STATS])
        
        for layer_idx in self.target_layers:
            for stat_name in stats_in_order:
                for head_idx in range(self.num_heads):
                    names.append(f"L{layer_idx}_{stat_name}_H{head_idx}")
        
        return names
    
    def get_mha_feature_names(self) -> List[str]:
        """Get names for MHA output features."""
        names = []
        for layer_idx in self.target_layers:
            for dim_idx in range(self.hidden_size):
                names.append(f"L{layer_idx}_mha_dim{dim_idx}")
        return names
    
    def __repr__(self):
        return (f"AttentionExtractor(layers={self.target_layers}, "
                f"stats={self.stats_to_compute}, "
                f"mha_output={self.extract_mha_output})")