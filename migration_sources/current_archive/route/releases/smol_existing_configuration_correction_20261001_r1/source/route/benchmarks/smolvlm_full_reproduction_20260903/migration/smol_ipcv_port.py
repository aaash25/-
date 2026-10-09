# Publication copy: comments/line endings only; algorithm AST unchanged.
# Project migration copy; retain upstream method and host notices. No experiment executed.
# Component licensing and exact source hashes: sources/FILE_DISPOSITIONS.json.
"""SmolVLM vision-only IPCV port: pruning + NGR + Attention Stabilization.

The author InternVL implementation is the crop-batch numerical reference under
third_party/ipcv_official_29726f0. Selection and attention remain per crop;
Neighbor-Guided Reconstruction searches all retained tokens in the encoder
batch, as in the author implementation. No DART/LLM pruning is enabled here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from smol_merge_ports import gather_rows


@dataclass
class Reconstruction:
    original_tokens: int
    kept: torch.Tensor
    removed: torch.Tensor
    original_kept: torch.Tensor
    removed_states: torch.Tensor
    neighbors: torch.Tensor  # [crop, removed slot, neighbor] -> global retained B*K index


@torch.no_grad()
def prune_and_save(current, previous, keep_ratio, top_k):
    batch, tokens, width = current.shape
    count = int(tokens * keep_ratio)
    if not 0 < count < tokens or top_k < 1:
        raise ValueError("IPCV active pruning requires 0 < keep count < tokens and top_k >= 1")
    scores = torch.norm(current - previous, dim=-1)
    kept = scores.topk(count, dim=1).indices.sort(dim=1).values
    ids = torch.arange(tokens, device=current.device).expand(batch, -1)
    kept_mask = torch.zeros((batch, tokens), device=current.device, dtype=torch.bool)
    kept_mask.scatter_(1, kept, True)
    removed = ids.masked_fill(kept_mask, tokens).sort(dim=1).values[:, :tokens-count]
    original_kept = gather_rows(current, kept)
    removed_states = gather_rows(current, removed)
    # Author InternVL keeps per-crop selection but searches one global B*K pool.
    # Preserve per-crop cdist calls as well as its FP32 distance arithmetic.
    all_kept = original_kept.reshape(batch * count, width)
    neighbors = torch.stack([
        torch.cdist(row.float(), all_kept.float(), p=2.0).topk(
            min(top_k, batch * count), largest=False, dim=-1
        ).indices
        for row in removed_states
    ])
    return original_kept, Reconstruction(tokens, kept, removed, original_kept, removed_states, neighbors)


def restore(current_kept, saved):
    batch, _, width = current_kept.shape
    delta = current_kept - saved.original_kept
    # Neighbor indices address the same global retained pool used at pruning.
    selected = delta.reshape(-1, width)[saved.neighbors]
    mean_delta = selected.mean(2)
    output = torch.zeros((batch, saved.original_tokens, width), device=current_kept.device, dtype=current_kept.dtype)
    output.scatter_(1, saved.kept[..., None].expand(-1, -1, width), current_kept)
    output.scatter_(1, saved.removed[..., None].expand(-1, -1, width), saved.removed_states + mean_delta)
    return output


class SmolIPCVPort:
    def __init__(self, encoder, *, prune_layer: int, keep_ratio: float, as_layers: int, top_k: int):
        if not 2 <= prune_layer < len(encoder.layers):
            raise ValueError("IPCV prune_layer is zero-based and must have a preceding layer delta")
        if not 0 < keep_ratio <= 1 or as_layers < 0 or top_k < 1:
            raise ValueError("Invalid IPCV configuration")
        self.encoder = encoder
        self.prune_layer, self.keep_ratio = prune_layer, keep_ratio
        self.as_layers, self.top_k = as_layers, top_k
        self.original_forward = None
        self.trace = []
        self.last_kept_indices = None

    def __enter__(self):
        import transformers
        if transformers.__version__ != "4.57.6":
            raise RuntimeError("This port has only been checked against transformers 4.57.6")
        if self.encoder.training or self.encoder.config._attn_implementation != "eager":
            raise RuntimeError("Migration contract requires eval mode and eager attention")
        if getattr(self.encoder, "_peer_migration_owner", None) is not None:
            raise RuntimeError("An encoder port is already active")
        self.original_forward = self.encoder.forward
        self.encoder._peer_migration_owner = self
        self.encoder.forward = self.forward
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.original_forward is not None:
            self.encoder.forward = self.original_forward
            del self.encoder._peer_migration_owner
            self.original_forward = None

    @torch.no_grad()
    def forward(self, inputs_embeds, attention_mask=None):
        from transformers.modeling_outputs import BaseModelOutput
        if self.original_forward is None:
            raise RuntimeError("Use SmolIPCVPort as a context manager")
        self.trace, self.last_kept_indices = [], None
        if self.keep_ratio == 1.0:
            return self.original_forward(inputs_embeds, attention_mask)
        if attention_mask is not None and bool(torch.any(attention_mask != 0)):
            raise NotImplementedError("Masked image patches require a separately verified port path")
        hidden, previous, saved = inputs_embeds, None, None
        for index, layer in enumerate(self.encoder.layers):
            if index == self.prune_layer - 1:
                previous = hidden
            if index == self.prune_layer:
                hidden, saved = prune_and_save(hidden, previous, self.keep_ratio, self.top_k)
                self.last_kept_indices = saved.kept
            stabilization = saved is not None and index < self.prune_layer + self.as_layers
            attended_input = restore(hidden, saved) if stabilization else hidden
            attended, _ = layer.self_attn(layer.layer_norm1(attended_input), attention_mask=None)
            hidden = attended_input + attended
            if stabilization:
                hidden = gather_rows(hidden, saved.kept)
            mlp_tokens = hidden.shape[1]
            hidden = hidden + layer.mlp(layer.layer_norm2(hidden))
            self.trace.append({"layer_zero_based": index, "attention_tokens": attended_input.shape[1],
                               "mlp_tokens": mlp_tokens, "attention_stabilization": stabilization})
        return BaseModelOutput(last_hidden_state=restore(hidden, saved))
