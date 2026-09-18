"""
LLM inference class based on Transformers library. 

Given a prompt, this module performs a single forward pass through 
a causal LM backbone and extracts last layer's hidden states.
Only need to finish Prefill stage.

A new method 'Model.generate': for each prompt, generate full length and return its reponse_ids_length 
"""

from __future__ import annotations

import gc
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import get_logger, resolve_device, torch_dtype_from_str


logger = get_logger(__name__)

class Model:
    """Extract final-layer, last-prompt-token representations from a causal LLM."""

    def __init__(
        self,
        model_id_or_path: str,
        batch_size: int = 16,
        device: str | torch.device = "auto",
        torch_dtype: str | torch.dtype = "bfloat16",
        max_prompt_length: int | None = None,
        trust_remote_code: bool = True,
        model: Any | None = None,
        tokenizer: Any | None = None,
        device_map: str | None = None,
        max_memory: dict[int | str, str] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_prompt_length is not None and max_prompt_length <= 0:
            raise ValueError("max_prompt_length must be positive when provided.")
        if (model is None) != (tokenizer is None):
            raise ValueError("model and tokenizer must be provided together.")

        self.model_id_or_path = model_id_or_path
        if max_memory is not None and device_map is None:
            raise ValueError("max_memory requires device_map.")
        self.device_map = device_map
        self.max_memory = max_memory
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self.torch_dtype = torch_dtype_from_str(torch_dtype)
        self.max_prompt_length = max_prompt_length
        self.trust_remote_code = trust_remote_code
        self.model = model
        self.tokenizer = tokenizer

        if self.model is not None:
            if not getattr(self.model, "hf_device_map", None):
                self.model.to(self.device)
            self.model.eval()

    def load_model(self):
        """Load the tokenizer and causal language model if they are not loaded."""
        if self.model is not None and self.tokenizer is not None:
            return self

        logger.info("Loading backbone %s on %s", self.model_id_or_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id_or_path,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither a pad token nor an EOS token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        loading_kwargs = {}
        if self.device_map is not None:
            loading_kwargs.update(device_map=self.device_map, max_memory=self.max_memory)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id_or_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
            **loading_kwargs,
        )
        if self.device_map is None:
            self.model.to(self.device)
        else:
            logger.info("Loaded model device map: %s", self.model.hf_device_map)
        self.model.eval()
        return self

    def _input_device(self) -> torch.device:
        """Place inputs on the embedding device of a dispatched model."""
        if getattr(self.model, "hf_device_map", None):
            embedding = self.model.get_input_embeddings()
            hook_device = getattr(getattr(embedding, "_hf_hook", None), "execution_device", None)
            if hook_device is not None:
                if isinstance(hook_device, int):
                    return torch.device(self.device.type, hook_device)
                return torch.device(hook_device)
            if embedding.weight.device.type != "meta":
                return embedding.weight.device
        return self.device

    @staticmethod
    def _last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
        """Find the final non-padding position for both left and right padding."""
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape (batch, sequence).")
        if torch.any(attention_mask.sum(dim=1) == 0):
            raise ValueError("Each prompt must contain at least one non-padding token.")
        # each sequence length is seq_len but with right padding tokens
        positions = torch.arange(attention_mask.shape[1], device=attention_mask.device) # [seq_len]
        positions = positions.unsqueeze(0).expand_as(attention_mask) # -> [1, seq_len] -> [batch_size, seq_len], [i,j] for each index
        return positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values 

    @torch.no_grad()
    def _extract_batch(self, prompts: Sequence[str]) -> torch.Tensor:
        assert self.model is not None and self.tokenizer is not None
        tokenizer_kwargs: dict[str, Any] = {
            "padding": True,
            "return_tensors": "pt",
            "add_special_tokens": True,
        }
        if self.max_prompt_length is not None:
            tokenizer_kwargs.update(
                truncation=True,
                max_length=self.max_prompt_length,
            )
        else:
            tokenizer_kwargs["truncation"] = False

        encoded = self.tokenizer(list(prompts), **tokenizer_kwargs)
        input_device = self._input_device()
        input_ids = encoded["input_ids"].to(input_device)
        attention_mask = encoded["attention_mask"].to(input_device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        if not outputs.hidden_states:
            raise RuntimeError("The backbone did not return hidden states.")
        hidden_states = outputs.hidden_states[-1] # last layer's hidden state [batch, seq_len, hidden_size]
        last_indices = self._last_token_indices(attention_mask) # last real token's position, exclude padding token [batch_size,]
        last_indices = last_indices.to(hidden_states.device)
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        features = hidden_states[batch_indices, last_indices] # extract target hidden states [batch_size, hidden_size]
        return features.detach().cpu().float()

    def extract(self, x: str | Sequence[str]) -> torch.Tensor:
        """Run prefill and return a tensor of shape ``(N, hidden_size)``."""
        prompts = [x] if isinstance(x, str) else list(x)
        if not prompts:
            raise ValueError("At least one prompt is required.")
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise TypeError("All prompts must be strings.")
        self.load_model()

        features = []
        for start in range(0, len(prompts), self.batch_size):
            features.append(self._extract_batch(prompts[start : start + self.batch_size]))
        return torch.cat(features, dim=0)

    @staticmethod
    def _count_response_tokens(
        response_ids: torch.Tensor,
        *,
        eos_token_id: int | Sequence[int] | None,
        pad_token_id: int | None,
    ) -> torch.Tensor:
        """Count generated tokens, excluding the terminating EOS and trailing padding."""
        if response_ids.ndim != 2:
            raise ValueError("response_ids must have shape (batch, generated_sequence).")
        if eos_token_id is None:
            eos_ids: set[int] = set()
        elif isinstance(eos_token_id, int):
            eos_ids = {eos_token_id}
        else:
            eos_ids = {int(token_id) for token_id in eos_token_id}

        lengths: list[int] = []
        for row in response_ids.detach().cpu().tolist():
            length = len(row)
            for index, token_id in enumerate(row):
                if token_id in eos_ids:
                    length = index
                    break
            else:
                if pad_token_id is not None:
                    while length > 0 and row[length - 1] == pad_token_id:
                        length -= 1
            lengths.append(length)
        return torch.tensor(lengths, dtype=torch.long)

    @torch.no_grad()
    def generate(
        self,
        x: str | Sequence[str],
        *,
        max_new_tokens: int | None = 8192,
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate complete responses and return their generated-token lengths.

        Lengths exclude prompt tokens, padding tokens, and the terminating EOS
        token. Generation stops at EOS or at ``max_new_tokens``. A tensor with
        shape ``(N,)`` is returned for both single and batched input.
        """
        prompts = [x] if isinstance(x, str) else list(x)
        if not prompts:
            raise ValueError("At least one prompt is required.")
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise TypeError("All prompts must be strings.")
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive or None.")
        if int(generation_kwargs.get("num_return_sequences", 1)) != 1:
            raise ValueError("generate() currently requires num_return_sequences=1.")
        if "input_ids" in generation_kwargs or "attention_mask" in generation_kwargs:
            raise ValueError("input_ids and attention_mask are constructed internally.")

        self.load_model()
        assert self.model is not None and self.tokenizer is not None
        original_padding_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        all_lengths: list[torch.Tensor] = []

        try:
            for start in range(0, len(prompts), self.batch_size):
                batch = prompts[start : start + self.batch_size]
                tokenizer_kwargs: dict[str, Any] = {
                    "padding": True,
                    "return_tensors": "pt",
                    "add_special_tokens": True,
                }
                if self.max_prompt_length is not None:
                    tokenizer_kwargs.update(
                        truncation=True,
                        max_length=self.max_prompt_length,
                    )
                else:
                    tokenizer_kwargs["truncation"] = False

                encoded = self.tokenizer(batch, **tokenizer_kwargs)
                input_device = self._input_device()
                input_ids = encoded["input_ids"].to(input_device)
                attention_mask = encoded["attention_mask"].to(input_device)
                call_kwargs = dict(generation_kwargs)
                if max_new_tokens is not None and "max_length" not in call_kwargs:
                    call_kwargs.setdefault("max_new_tokens", max_new_tokens)
                call_kwargs.setdefault("do_sample", False)
                call_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
                if self.tokenizer.eos_token_id is not None:
                    call_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **call_kwargs,
                )
                sequences = generated.sequences if hasattr(generated, "sequences") else generated
                if sequences.ndim != 2 or sequences.shape[0] != len(batch):
                    raise RuntimeError(
                        "Backbone generate() returned an unexpected sequence tensor shape."
                    )
                response_ids = sequences[:, input_ids.shape[1] :]
                all_lengths.append(
                    self._count_response_tokens(
                        response_ids,
                        eos_token_id=call_kwargs.get("eos_token_id"),
                        pad_token_id=call_kwargs.get("pad_token_id"),
                    )
                )
        finally:
            self.tokenizer.padding_side = original_padding_side

        return torch.cat(all_lengths, dim=0)

    def unload_model(self) -> None:
        """Release the loaded backbone and tokenizer."""
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "npu") and torch.npu.is_available():
            for index in range(torch.npu.device_count()):
                with torch.npu.device(index):
                    torch.npu.empty_cache()

    close = unload_model
