# Exact source-node selection for publication; retained implementations are unchanged.
# iLLaVA helper omitted because its root license is unresolved. SmolMergePort host code remains unchanged; iLLaVA mode requires the omitted helper and is unsupported by this selected component.
# PiToMe portions CC-BY-NC-4.0; DyMU-related portions MIT; host portions preserve HF Apache notices.
"""Development ports of the author PiToMe HF and iLLaVA SigLIP vision paths.

No training, model loading, GPU job, or benchmark is triggered by importing this
module. The only added model-boundary operation is restoration of original patch
slots before SmolVLM's unchanged post-layernorm and pixel-shuffle connector.
Sources and differences are recorded in SOURCE_NOTES.md.
"""


from __future__ import annotations


import math


from dataclasses import dataclass


import torch


import torch.nn.functional as F


@dataclass
class MergeStep:
    hidden: torch.Tensor
    parent: torch.Tensor  # [crop, old slot] -> compact slot


def gather_rows(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return x.gather(1, index[..., None].expand(-1, -1, x.shape[-1]))


def identity_parent(x: torch.Tensor) -> torch.Tensor:
    return torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)


def _assemble(x, protected_idx, target_idx, source_idx, destination):
    """Author mean reduction, with a compact integer restoration map."""
    batch, tokens, channels = x.shape
    protected = gather_rows(x, protected_idx)
    targets = gather_rows(x, target_idx)
    sources = gather_rows(x, source_idx)
    targets = targets.scatter_reduce(
        1, destination[..., None].expand(-1, -1, channels), sources, reduce="mean"
    )
    output = torch.cat((protected, targets), dim=1)
    offset = protected_idx.shape[1]
    parent = torch.empty((batch, tokens), device=x.device, dtype=torch.long)
    parent.scatter_(1, protected_idx, torch.arange(offset, device=x.device).expand(batch, -1))
    parent.scatter_(
        1, target_idx,
        (offset + torch.arange(target_idx.shape[1], device=x.device)).expand(batch, -1),
    )
    parent.scatter_(1, source_idx, offset + destination)
    return MergeStep(output, parent)


@torch.no_grad()
def pitome_step(x: torch.Tensor, ratio: float, margin: float, use_bsm: bool) -> MergeStep:
    """No-CLS author HF-CLIP branch: mean merge, not size-weighted merge."""
    if not 0.5 <= ratio <= 1.0:
        raise ValueError("PiToMe per-layer retained ratio must be in [0.5, 1]")
    batch, tokens, _ = x.shape
    removed = math.floor(tokens - tokens * ratio)
    if removed == 0:
        return MergeStep(x, identity_parent(x))
    if use_bsm:
        metric = x / x.norm(dim=-1, keepdim=True)
        scores = metric[:, ::2] @ metric[:, 1::2].transpose(-1, -2)
        maximum, destination = scores.max(dim=-1)
        order = maximum.argsort(dim=-1, descending=True)
        sources = order[:, :removed]
        protected_idx = 2 * order[:, removed:]
        target_idx = torch.arange(1, tokens, 2, device=x.device).expand(batch, -1)
        return _assemble(x, protected_idx, target_idx, 2 * sources, destination.gather(1, sources))
    metric = F.normalize(x, p=2, dim=-1)
    similarity = metric @ metric.transpose(-1, -2)
    energy = F.elu(similarity - margin, alpha=1.0).mean(dim=-1)
    order = energy.argsort(dim=-1, descending=True)
    merge_idx = order[:, :2 * removed]
    protected_idx = order[:, 2 * removed:]
    source_idx, target_idx = merge_idx[:, ::2], merge_idx[:, 1::2]
    scores = similarity.gather(2, target_idx[:, None].expand(batch, tokens, removed))
    scores = scores.gather(1, source_idx[..., None].expand(batch, removed, removed))
    destination = scores.max(dim=-1).indices
    return _assemble(x, protected_idx, target_idx, source_idx, destination)


class SmolMergePort:
    """Temporary encoder-only patch for transformers 4.57.6 SmolVLM.

    The current development contract covers fully valid image crops. Masked
    patches explicitly fail instead of silently changing attention semantics.
    Real-image GPU validation must establish that the intended workload fits
    this contract, or masked-crop support must be added before formal evaluation.
    """

    def __init__(self, encoder, method: str, *, ratio=1.0, merge_layers=(), removed=0):
        if method not in ("pitome", "illava_vit"):
            raise ValueError(method)
        if method == "pitome" and not 0.5 <= ratio <= 1.0:
            raise ValueError("ratio must be in [0.5, 1]")
        self.encoder, self.method = encoder, method
        self.ratio, self.merge_layers, self.removed = ratio, frozenset(merge_layers), removed
        if removed < 0 or any(i < 0 or i >= len(encoder.layers) for i in self.merge_layers):
            raise ValueError("invalid zero-based merge schedule")
        self.original_forward = None
        self.trace = []
        self.last_parent = None

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

    def close(self):
        if self.original_forward is not None:
            self.encoder.forward = self.original_forward
            del self.encoder._peer_migration_owner
            self.original_forward = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @torch.no_grad()
    def forward(self, inputs_embeds, attention_mask=None):
        from transformers.modeling_outputs import BaseModelOutput
        if self.original_forward is None:
            raise RuntimeError("Use SmolMergePort as a context manager")
        self.trace, self.last_parent = [], None  # reset on every image batch
        disabled = (self.method == "pitome" and self.ratio == 1.0) or (
            self.method == "illava_vit" and (not self.merge_layers or self.removed == 0)
        )
        if disabled:
            return self.original_forward(inputs_embeds, attention_mask)
        if attention_mask is not None and bool(torch.any(attention_mask != 0)):
            raise NotImplementedError("Masked image patches require a separately verified port path")
        hidden = inputs_embeds
        parent = identity_parent(hidden)
        count = len(self.encoder.layers)
        for index, layer in enumerate(self.encoder.layers):
            before = hidden.shape[1]
            if self.method == "pitome":
                hidden = layer(hidden, None)
                if not isinstance(hidden, torch.Tensor):
                    raise TypeError("Unexpected SmolVLM layer return type")
                # Author HF implementation: idx <= 12 uses BSM, then PiToMe.
                step = pitome_step(hidden, self.ratio, 0.9 - 0.9 * index / count, index <= 12)
                hidden = step.hidden
                parent = step.parent.gather(1, parent)
                mlp_tokens = before
            else:
                attended, attention = layer.self_attn(layer.layer_norm1(hidden), attention_mask=None)
                hidden = hidden + attended
                if index in self.merge_layers:
                    step = illava_step(hidden, attention, self.removed)
                    hidden = step.hidden
                    parent = step.parent.gather(1, parent)
                mlp_tokens = hidden.shape[1]
                hidden = hidden + layer.mlp(layer.layer_norm2(hidden))
            self.trace.append({"layer_zero_based": index, "attention_tokens": before,
                               "mlp_tokens": mlp_tokens, "output_tokens": hidden.shape[1]})
        self.last_parent = parent
        return BaseModelOutput(last_hidden_state=gather_rows(hidden, parent))
