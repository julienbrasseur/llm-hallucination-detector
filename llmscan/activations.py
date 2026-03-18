"""
Activation extraction utilities for token-wise or pooled hidden activations
from any Hugging Face chat model.

This script exposes ActivationExtractor with a flexible `extract` API that
can:
 - return mean-pooled per-sample activations (existing behavior), OR
 - stream token-wise activations and write them into disk shards
   (recommended for large corpora).

Features:
 - automatic device_map="auto" model loading
 - **model-agnostic** assistant masking (works with Mistral, Llama, Qwen,
   Gemma, and any model with a HF chat template)
 - configurable shard size (in tokens) and dtype for disk storage
 - efficient batched extraction with minimal GPU memory retention

Assistant masking strategy (selected at init, reported to user):
 1. Native: apply_chat_template(..., return_assistant_tokens_mask=True)
 2. Response template matching: auto-detect the assistant turn delimiter
    from the chat template and search for it in tokenized output
    (inspired by trl's DataCollatorForCompletionOnlyLM, no dependency)
 3. Prefix diff: tokenize full conv vs user-only prefix, diff = assistant
 4. Last resort: mark all non-padding tokens (with warning)
"""

from typing import List, Dict, Any, Optional, Union
import os
import warnings
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Masking strategy names (for reporting)
_STRATEGY_NATIVE = "native (return_assistant_tokens_mask)"
_STRATEGY_RESPONSE_TEMPLATE = "response template matching"
_STRATEGY_PREFIX = "prefix diff (tokenize twice and compare)"
_STRATEGY_FALLBACK = "fallback (all tokens marked as assistant)"


class ActivationExtractor:
    def __init__(
        self,
        model_name: str,
        target_layers: List[int],
        device: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Load model+tokenizer with device_map='auto' and set parameters.

        Args:
            model_name: HF model identifier (any model with a chat template)
            target_layers: list of layer indices (0-based) to extract
            device: preferred CUDA device for inputs (e.g. 'cuda:0'). Model
                    itself is loaded with device_map='auto'.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, **(tokenizer_kwargs or {})
        )
        self.target_layers = target_layers

        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif getattr(self.tokenizer, "unk_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.tokenizer.padding_side = "right"

        # Detect and select the best assistant masking strategy
        self._masking_strategy, self._response_template_ids = (
            self._select_masking_strategy()
        )

        print(
            f"Loaded model {model_name}.\n"
            f"  Target layers: {target_layers}\n"
            f"  Device: {self.device}\n"
            f"  Assistant masking: {self._masking_strategy}"
        )

    # ------------------------------------------------------------------
    # Masking strategy selection (run once at init)
    # ------------------------------------------------------------------
    def _select_masking_strategy(self):
        """Probe the tokenizer and select the best masking strategy.

        Returns (strategy_name, response_template_ids_or_None).
        """
        # Strategy 1: Native return_assistant_tokens_mask
        if self._check_native_assistant_mask():
            return _STRATEGY_NATIVE, None

        # Strategy 2: Response template matching (trl-inspired, no dependency)
        if hasattr(self.tokenizer, "apply_chat_template"):
            template_ids = self._detect_response_template()
            if template_ids is not None:
                return _STRATEGY_RESPONSE_TEMPLATE, template_ids

        # Strategy 3: Prefix diff
        if hasattr(self.tokenizer, "apply_chat_template"):
            return _STRATEGY_PREFIX, None

        # Strategy 4: Fallback
        return _STRATEGY_FALLBACK, None

    def _check_native_assistant_mask(self) -> bool:
        """Check if tokenizer supports return_assistant_tokens_mask."""
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return False
        test_conv = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "reply"},
        ]
        try:
            result = self.tokenizer.apply_chat_template(
                test_conv,
                return_assistant_tokens_mask=True,
                tokenize=True,
            )
            result = self._normalize_chat_template_output(result)
            if isinstance(result, dict) and "assistant_masks" in result:
                # Verify the mask actually has nonzero entries
                mask = result["assistant_masks"]
                if any(m != 0 for m in mask):
                    return True
                return False  # mask is all zeros — not functional
            if isinstance(result, (list, tuple)) and len(result) >= 2:
                mask = result[1]
                if any(m != 0 for m in mask):
                    return True
                return False
        except (TypeError, Exception):
            return False
        return False

    @staticmethod
    def _normalize_chat_template_output(result):
        """Normalize the output of apply_chat_template(tokenize=True).

        Different tokenizers return different types:
        - list of ints (standard)
        - dict with "input_ids" key
        - BatchEncoding with .input_ids attribute
        We normalize to a plain list of ints, or a dict with list values.
        """
        # BatchEncoding or dict-like with input_ids
        if hasattr(result, "input_ids") and not isinstance(result, (list, tuple)):
            ids = result["input_ids"]
            # Might be batched: [[ids...]]
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            # Check if there's an assistant mask too
            if hasattr(result, "keys") and "assistant_masks" in result:
                masks = result["assistant_masks"]
                if isinstance(masks, list) and masks and isinstance(masks[0], list):
                    masks = masks[0]
                return {"input_ids": ids, "assistant_masks": masks}
            return ids

        # Plain dict
        if isinstance(result, dict):
            ids = result.get("input_ids", result)
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            if "assistant_masks" in result:
                masks = result["assistant_masks"]
                if isinstance(masks, list) and masks and isinstance(masks[0], list):
                    masks = masks[0]
                return {"input_ids": ids, "assistant_masks": masks}
            result["input_ids"] = ids
            return result

        # Already a plain list
        return result

    def _detect_response_template(self) -> Optional[List[int]]:
        """Auto-detect the response template token IDs from the chat template.

        Generates the template for a minimal conversation and extracts
        the token sequence that marks the start of the assistant response.
        """
        try:
            # Build a conversation where we can identify the assistant boundary
            prefix_conv = [{"role": "user", "content": "test"}]
            prefix_text = self.tokenizer.apply_chat_template(
                prefix_conv, tokenize=False, add_generation_prompt=True
            )
            # The generation prompt is the text the model would see right before
            # generating the assistant response — this IS the response template
            user_only_text = self.tokenizer.apply_chat_template(
                prefix_conv, tokenize=False, add_generation_prompt=False
            )
            # The response template = generation_prompt - user_only
            response_template_str = prefix_text[len(user_only_text):]
            if not response_template_str.strip():
                return None

            response_template_ids = self.tokenizer.encode(
                response_template_str, add_special_tokens=False
            )
            if not response_template_ids:
                return None

            return response_template_ids
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Core: get assistant mask for a single conversation
    # ------------------------------------------------------------------
    def _get_assistant_mask_for_ids(
        self,
        input_ids: List[int],
        conv: List[Dict[str, str]],
    ) -> List[int]:
        """Return a binary mask (list of 0/1) over input_ids marking
        assistant tokens, using the selected strategy.
        """
        strategy = self._masking_strategy

        # --- Strategy 1: Native ---
        if strategy == _STRATEGY_NATIVE:
            try:
                result = self.tokenizer.apply_chat_template(
                    conv,
                    return_assistant_tokens_mask=True,
                    tokenize=True,
                )
                result = self._normalize_chat_template_output(result)
                if isinstance(result, dict):
                    return result["assistant_masks"]
                else:
                    return result[1]
            except Exception:
                pass  # fall through

        # --- Strategy 2: Response template matching ---
        if strategy == _STRATEGY_RESPONSE_TEMPLATE and self._response_template_ids is not None:
            mask = self._response_template_mask(input_ids, self._response_template_ids)
            if mask is not None:
                return mask

        # --- Strategy 3: Prefix diff ---
        if strategy in (_STRATEGY_PREFIX, _STRATEGY_RESPONSE_TEMPLATE):
            # response template might have failed on this example; try prefix diff
            mask = self._prefix_diff_mask(input_ids, conv)
            if mask is not None:
                return mask

        # --- Strategy 4: Fallback ---
        return [1] * len(input_ids)

    def _response_template_mask(
        self, input_ids: List[int], response_template_ids: List[int]
    ) -> Optional[List[int]]:
        """Find the response template in input_ids and mark everything
        after it as assistant tokens.
        """
        template_len = len(response_template_ids)
        # Search for the LAST occurrence of the response template
        last_match = -1
        for i in range(len(input_ids) - template_len + 1):
            if input_ids[i : i + template_len] == response_template_ids:
                last_match = i

        if last_match == -1:
            return None  # template not found

        assistant_start = last_match + template_len
        mask = [0] * len(input_ids)
        for i in range(assistant_start, len(input_ids)):
            mask[i] = 1
        return mask

    def _prefix_diff_mask(
        self, input_ids: List[int], conv: List[Dict[str, str]]
    ) -> Optional[List[int]]:
        """Compute assistant mask via prefix diff: tokenize user-only
        prefix and mark everything after it as assistant.
        """
        try:
            last_assistant_idx = None
            for idx in range(len(conv) - 1, -1, -1):
                if conv[idx]["role"] == "assistant":
                    last_assistant_idx = idx
                    break

            if last_assistant_idx is None:
                return None

            prefix_conv = conv[:last_assistant_idx]
            prefix_text = self.tokenizer.apply_chat_template(
                prefix_conv, tokenize=False, add_generation_prompt=True
            )
            prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
            prefix_len = len(prefix_ids)

            mask = [0] * len(input_ids)
            for i in range(prefix_len, len(input_ids)):
                mask[i] = 1
            return mask
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Batch tokenization with assistant masks
    # ------------------------------------------------------------------
    def _tokenize_conversations(
        self,
        conversations: List[List[Dict[str, str]]],
        max_length: int,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize conversations and return input_ids, attention_mask,
        and assistant_mask tensors.
        """
        all_input_ids = []
        all_assistant_masks = []

        for conv in conversations:
            # Tokenize full conversation
            full_ids = self.tokenizer.apply_chat_template(
                conv, tokenize=True, add_generation_prompt=False,
                max_length=max_length, truncation=True,
            )
            full_ids = self._normalize_chat_template_output(full_ids)
            # If normalize returned a dict (from native mask), extract just ids
            if isinstance(full_ids, dict):
                full_ids = full_ids["input_ids"]

            # Get assistant mask using the selected strategy
            assistant_mask = self._get_assistant_mask_for_ids(full_ids, conv)

            # Truncate mask to match ids length (in case of mismatch)
            assistant_mask = assistant_mask[: len(full_ids)]
            if len(assistant_mask) < len(full_ids):
                assistant_mask.extend([0] * (len(full_ids) - len(assistant_mask)))

            all_input_ids.append(torch.tensor(full_ids, dtype=torch.long))
            all_assistant_masks.append(torch.tensor(assistant_mask, dtype=torch.long))

        # Pad to max length in batch
        max_len = max(ids.size(0) for ids in all_input_ids)
        batch_size = len(all_input_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        padded_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        assistant_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i, (ids, amask) in enumerate(zip(all_input_ids, all_assistant_masks)):
            length = ids.size(0)
            padded_ids[i, :length] = ids
            attention_mask[i, :length] = 1
            assistant_mask[i, :length] = amask

        return {
            "input_ids": padded_ids,
            "attention_mask": attention_mask,
            "assistant_mask": assistant_mask,
        }

    # ------------------------------------------------------------------
    # Data formatting (backward-compatible)
    # ------------------------------------------------------------------
    def _format_conversation(self, item: Union[Dict[str, Any], str]) -> str:
        """Accept either a preformatted conversation dict or a raw string."""
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
            parts = []
            for turn in conv:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                parts.append(f"<{role}>: {content}")
            return "\n".join(parts)

        raise ValueError("Unsupported item type for formatting")

    @staticmethod
    def _get_conversation(
        item: Union[Dict[str, Any], str],
    ) -> Optional[List[Dict[str, str]]]:
        """Extract the conversation list from an item, or None if raw string."""
        if isinstance(item, dict) and "conversation" in item:
            return item["conversation"]
        return None

    # ------------------------------------------------------------------
    # Extraction APIs
    # ------------------------------------------------------------------
    def extract(
        self,
        raw_texts: List[Union[Dict[str, Any], str]],
        batch_size: int = 8,
        max_length: int = 1024,
        mean_pool: bool = True,
        focus_on_assistant: bool = False,
    ) -> torch.Tensor:
        """Extract activations for provided examples and return a single tensor.

        When mean_pool=True we return a tensor shaped [N, num_layers * hidden].
        """
        all_acts = []

        for i in tqdm(range(0, len(raw_texts), batch_size), desc="Extracting"):
            batch_items = raw_texts[i : i + batch_size]

            if focus_on_assistant:
                conversations = [self._get_conversation(item) for item in batch_items]
                has_convs = any(c is not None for c in conversations)

                if has_convs:
                    valid_convs = [c for c in conversations if c is not None]
                    tokenized = self._tokenize_conversations(valid_convs, max_length)
                    inputs = {
                        "input_ids": tokenized["input_ids"].to(self.device),
                        "attention_mask": tokenized["attention_mask"].to(self.device),
                    }
                    mask = tokenized["assistant_mask"].to(self.device)
                else:
                    texts = [self._format_conversation(x) for x in batch_items]
                    inputs = self.tokenizer(
                        texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length,
                    ).to(self.device)
                    mask = inputs["attention_mask"]
            else:
                texts = [self._format_conversation(x) for x in batch_items]
                inputs = self.tokenizer(
                    texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length,
                ).to(self.device)
                mask = inputs["attention_mask"]

            with torch.no_grad():
                outputs = self.model(**inputs)

            hidden_states = outputs.hidden_states

            if mean_pool:
                mask_exp = mask.unsqueeze(-1).to(hidden_states[0].dtype)
                mask_sum = mask_exp.sum(dim=1, keepdim=False)

                selected = torch.stack(
                    [hidden_states[l] for l in self.target_layers], dim=1
                )
                masked = selected * mask_exp.unsqueeze(1)
                pooled = masked.sum(dim=2) / (mask_sum.unsqueeze(1) + 1e-12)
                batch_acts = pooled.reshape(pooled.size(0), -1).cpu()
                all_acts.append(batch_acts)
            else:
                for l in self.target_layers:
                    layer_tensor = hidden_states[l].cpu()
                    mask_cpu = mask.cpu()
                    for b in range(layer_tensor.size(0)):
                        seq_mask = mask_cpu[b].bool()
                        n_tokens = int(seq_mask.sum().item())
                        if n_tokens == 0:
                            continue
                        token_acts = layer_tensor[b, seq_mask, :].clone()
                        all_acts.append(token_acts)

            del outputs
            torch.cuda.empty_cache()

        if mean_pool:
            return torch.cat(all_acts, dim=0)
        else:
            return all_acts

    def extract_to_shards(
        self,
        raw_texts: List[Union[Dict[str, Any], str]],
        out_dir: str,
        shard_size_tokens: int = 100_000,
        dtype: torch.dtype = torch.float16,
        batch_size: int = 8,
        max_length: int = 1024,
        focus_on_assistant: bool = False,
        per_example: bool = False,
        shard_size_examples: int = 5_000,
    ) -> List[str]:
        """Stream token-wise activations and write to disk in shards.

        When per_example=False (default), shards contain flat concatenated
        token activations — backward-compatible with existing consumers.

        When per_example=True, each shard is a dict with:
          - "sequences": list of tensors, each shaped [seq_len_i, H]
          - "lengths": 1-D int tensor of per-example token counts
        This format is required for sequence-aware probes (e.g. TCNProbe).

        Returns list of shard file paths.
        """
        os.makedirs(out_dir, exist_ok=True)

        shard_idx = 0
        shard_paths = []

        def _get_inputs_and_mask(batch_items):
            """Tokenize a batch and return (inputs_dict, mask_tensor)."""
            if focus_on_assistant:
                conversations = [self._get_conversation(item) for item in batch_items]
                has_convs = any(c is not None for c in conversations)

                if has_convs:
                    valid_convs = [c for c in conversations if c is not None]
                    tokenized = self._tokenize_conversations(valid_convs, max_length)
                    inputs = {
                        "input_ids": tokenized["input_ids"].to(self.device),
                        "attention_mask": tokenized["attention_mask"].to(self.device),
                    }
                    mask = tokenized["assistant_mask"].to(self.device)
                    return inputs, mask

            texts = [self._format_conversation(x) for x in batch_items]
            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(self.device)
            return inputs, inputs["attention_mask"]

        if per_example:
            example_buffer: List[torch.Tensor] = []

            for i in tqdm(
                range(0, len(raw_texts), batch_size),
                desc="Shard extracting (per-example)",
            ):
                batch_items = raw_texts[i : i + batch_size]
                inputs, mask = _get_inputs_and_mask(batch_items)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                hidden_states = outputs.hidden_states

                for l in self.target_layers:
                    layer_tensor = hidden_states[l]
                    mask_cpu = mask.cpu()
                    layer_cpu = layer_tensor.cpu()
                    for b in range(layer_cpu.size(0)):
                        seq_mask = mask_cpu[b].bool()
                        n_tokens = int(seq_mask.sum().item())
                        if n_tokens == 0:
                            example_buffer.append(
                                torch.empty(0, layer_cpu.size(-1), dtype=dtype)
                            )
                            continue
                        token_acts = layer_cpu[b, seq_mask, :].to(dtype).clone()
                        example_buffer.append(token_acts)

                    if len(example_buffer) >= shard_size_examples:
                        lengths = torch.tensor(
                            [t.size(0) for t in example_buffer], dtype=torch.long
                        )
                        shard_path = os.path.join(
                            out_dir, f"acts_shard_{shard_idx:05d}.pt"
                        )
                        torch.save(
                            {"sequences": example_buffer, "lengths": lengths},
                            shard_path,
                        )
                        shard_paths.append(shard_path)
                        shard_idx += 1
                        example_buffer = []

                del outputs
                torch.cuda.empty_cache()

            if example_buffer:
                lengths = torch.tensor(
                    [t.size(0) for t in example_buffer], dtype=torch.long
                )
                shard_path = os.path.join(
                    out_dir, f"acts_shard_{shard_idx:05d}.pt"
                )
                torch.save(
                    {"sequences": example_buffer, "lengths": lengths},
                    shard_path,
                )
                shard_paths.append(shard_path)

        else:
            shard_tensors = []
            shard_token_count = 0

            for i in tqdm(
                range(0, len(raw_texts), batch_size), desc="Shard extracting"
            ):
                batch_items = raw_texts[i : i + batch_size]
                inputs, mask = _get_inputs_and_mask(batch_items)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                hidden_states = outputs.hidden_states

                for l in self.target_layers:
                    layer_tensor = hidden_states[l]
                    for b in range(layer_tensor.size(0)):
                        seq_mask = mask[b].bool()
                        n_tokens = int(seq_mask.sum().item())
                        if n_tokens == 0:
                            continue
                        token_acts = layer_tensor[b, seq_mask, :].cpu()

                        shard_tensors.append(token_acts)
                        shard_token_count += n_tokens

                        if shard_token_count >= shard_size_tokens:
                            merged = torch.cat(shard_tensors, dim=0).to(dtype)
                            shard_path = os.path.join(
                                out_dir, f"acts_shard_{shard_idx:05d}.pt"
                            )
                            torch.save(merged, shard_path)
                            shard_paths.append(shard_path)

                            shard_idx += 1
                            shard_tensors = []
                            shard_token_count = 0

                del outputs
                torch.cuda.empty_cache()

            if shard_tensors:
                merged = torch.cat(shard_tensors, dim=0).to(dtype)
                shard_path = os.path.join(
                    out_dir, f"acts_shard_{shard_idx:05d}.pt"
                )
                torch.save(merged, shard_path)
                shard_paths.append(shard_path)

        return shard_paths
