"""
Activation extraction utilities for token-wise or pooled hidden activations
from a Hugging Face transformer (Ministral-8B-Instruct-2410).

This script exposes ActivationExtractor with a flexible `extract` API that
can:
 - return mean-pooled per-sample activations (existing behavior), OR
 - stream token-wise activations and write them into disk shards
   (recommended for large corpora).

Features:
 - automatic device_map="auto" model loading
 - optional focus_on_assistant masking (keeps only assistant tokens)
 - configurable shard size (in tokens) and dtype for disk storage
 - efficient batched extraction with minimal GPU memory retention

Usage examples:
  extractor = ActivationExtractor(
      model_name="mistralai/Ministral-8B-Instruct-2410",
      target_layers=[19],
      device='cuda',
  )

  # write token-wise shards to disk
  extractor.extract_to_shards(
      dataset_texts,           # list of conversation dicts or raw strings
      out_dir="./shards",
      shard_size_tokens=100_000,
      focus_on_assistant=True,
      mean_pool=False
  )

  # quick pooled extraction
  acts = extractor.extract(dataset_texts[:1024], mean_pool=True)

Notes:
 - This file assumes the tokenizer exposes a sensible chat templating helper
   (some HF chat models provide tokenizer.apply_chat_template). If not, a
   simple concatenation fallback is used.
 - The assistant masking looks for an instruction boundary. We try to be
   tokenizer-agnostic but you may want to tweak _find_last_inst_token.
"""

from typing import List, Dict, Any, Optional, Generator, Union
import os
import math
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


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
            model_name: HF model identifier
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **(tokenizer_kwargs or {}))
        self.target_layers = target_layers

        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif getattr(self.tokenizer, "unk_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                # fallback to a sentinel
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.tokenizer.padding_side = "right"

        # Attempt to identify an instruction-end token used in chat templates.
        # This is heuristic; if your model uses a different sentinel, set
        # extractor.end_token_id manually.
        try:
            self.end_token_id = self.tokenizer.convert_tokens_to_ids("[/INST]")
            if self.end_token_id == self.tokenizer.unk_token_id:
                self.end_token_id = None
        except Exception:
            self.end_token_id = None

        print(f"Loaded model {model_name}. Target layers: {target_layers}. Device for inputs: {self.device}")

    # Data formatting
    def _format_conversation(self, item: Union[Dict[str, Any], str]) -> str:
        """Accept either a preformatted conversation dict or a raw string.

        Expected dict format:
            {"conversation": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}

        If tokenizer exposes `apply_chat_template` we prefer it, otherwise
        we fall back to a simple user/assistant concatenation.
        """
        if isinstance(item, str):
            return item

        if isinstance(item, dict) and "conversation" in item:
            conv = item["conversation"]
            # Try HF convenience method if present
            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    return self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
                except Exception:
                    pass

            # fallback: simple concatenation
            parts = []
            for turn in conv:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                parts.append(f"<{role}>: {content}")
            return "\n".join(parts)

        raise ValueError("Unsupported item type for formatting to conversation string")

    # Assistant masking
    def _create_assistant_mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return a mask with 1s on assistant tokens and 0 elsewhere.

        Heuristic: find last occurrence of an instruction separator (if known)
        and set assistant tokens after it. When no separator is known we assume
        the second half (after the first user token) is assistant — this is
        imperfect; prefer to supply chat template-concatenated conversations
        where assistant content follows user content.
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        out = torch.zeros_like(attention_mask)

        if self.end_token_id is not None:
            is_end = (input_ids == self.end_token_id)
            for b in range(batch_size):
                inst_pos = (is_end[b].nonzero(as_tuple=True)[0])
                if len(inst_pos) > 0:
                    start = inst_pos[-1].item() + 1
                    seq_length = int(attention_mask[b].sum().item())
                    if start < seq_length:
                        out[b, start:seq_length] = 1
                    else:
                        # nothing after tag, fallback to zeros
                        pass
                else:
                    # fallback: mark all non-padding tokens
                    seq_length = int(attention_mask[b].sum().item())
                    out[b, :seq_length] = 1
        else:
            # No reliable sentinel so fallback to simple heuristic: assume even tokens
            # after half sequence belong to assistant. This is a last-resort.
            for b in range(batch_size):
                seq_length = int(attention_mask[b].sum().item())
                if seq_length <= 1:
                    continue
                mid = seq_length // 2
                out[b, mid:seq_length] = 1

        return out

    # Extraction APIs
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
        texts = [self._format_conversation(x) for x in raw_texts]
        all_acts = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
            batch = texts[i : i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            hidden_states = outputs.hidden_states  # tuple: (layer0, layer1, ...), each [B, S, H]
            mask = inputs["attention_mask"]

            if focus_on_assistant:
                mask = self._create_assistant_mask(inputs["input_ids"], mask)

            if mean_pool:
                mask_exp = mask.unsqueeze(-1).to(hidden_states[0].dtype)
                mask_sum = mask_exp.sum(dim=1, keepdim=False)  # [B, 1]

                # stack selected layers
                selected = torch.stack([hidden_states[l] for l in self.target_layers], dim=1)  # [B, L, S, H]
                masked = selected * mask_exp.unsqueeze(1)  # [B, L, S, H]
                pooled = masked.sum(dim=2) / (mask_sum.unsqueeze(1) + 1e-12)  # [B, L, H]
                batch_acts = pooled.reshape(pooled.size(0), -1).cpu()
                all_acts.append(batch_acts)

            else:
                # Token-wise: yield per-token activation blocks for each sample in the batch
                for l in self.target_layers:
                    layer_tensor = hidden_states[l].cpu()
                    mask_cpu = mask.cpu()  # Move mask to CPU once
                    for b in range(layer_tensor.size(0)):
                        seq_mask = mask_cpu[b].bool()  # Now create mask from CPU tensor
                        n_tokens = int(seq_mask.sum().item())
                        if n_tokens == 0:
                            continue
                        token_acts = layer_tensor[b, seq_mask, :].clone()  # [n_tokens, H]
                        # Flattening choice: save token-wise rows and let caller assemble
                        all_acts.append(token_acts)

            # free memory
            del outputs
            torch.cuda.empty_cache()

        if mean_pool:
            final = torch.cat(all_acts, dim=0)
            return final
        else:
            # On token-wise mode we return a list of tensors (variable-length rows)
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
    ) -> List[str]:
        """Stream token-wise activations and write to disk in shards.

        Returns list of shard file paths.
        """
        os.makedirs(out_dir, exist_ok=True)
        texts = [self._format_conversation(x) for x in raw_texts]

        shard_idx = 0
        shard_tensors = []
        shard_token_count = 0
        shard_paths = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Shard extracting"):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            hidden_states = outputs.hidden_states
            mask = inputs["attention_mask"]

            if focus_on_assistant:
                mask = self._create_assistant_mask(inputs["input_ids"], mask)

            for l in self.target_layers:
                layer_tensor = hidden_states[l]  # [B, S, H]
                for b in range(layer_tensor.size(0)):
                    seq_mask = mask[b].bool()
                    n_tokens = int(seq_mask.sum().item())
                    if n_tokens == 0:
                        continue
                    token_acts = layer_tensor[b, seq_mask, :].cpu()  # [n_tokens, H]

                    shard_tensors.append(token_acts)
                    shard_token_count += n_tokens

                    if shard_token_count >= shard_size_tokens:
                        # concatenate and save shard
                        merged = torch.cat(shard_tensors, dim=0).to(dtype)
                        shard_path = os.path.join(out_dir, f"acts_shard_{shard_idx:05d}.pt")
                        torch.save(merged, shard_path)
                        shard_paths.append(shard_path)

                        shard_idx += 1
                        shard_tensors = []
                        shard_token_count = 0

            # cleanup
            del outputs
            torch.cuda.empty_cache()

        # save remainder
        if shard_tensors:
            merged = torch.cat(shard_tensors, dim=0).to(dtype)
            shard_path = os.path.join(out_dir, f"acts_shard_{shard_idx:05d}.pt")
            torch.save(merged, shard_path)
            shard_paths.append(shard_path)

        return shard_paths