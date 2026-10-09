# Publication copy: comments/line endings only; algorithm AST unchanged.
# Project migration copy; retain upstream method and host notices. No experiment executed.
# Component licensing and exact source hashes: sources/FILE_DISPOSITIONS.json.
"""Qwen FP16 eager text score compatibility for the common 4.57.6 stack.

Qwen's eager path computes QK in FP16 before its FP32 softmax.  A finite QK
dot product can overflow FP16 and turn the subsequent softmax into NaNs.  This
module changes only the QK score accumulation to FP32 for FP16 text attention.
Vision attention, value mixing, model weights, and the eager backend stay as is.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fp32_score_eager_attention(module, query, key, value, attention_mask,
                               scaling, dropout=0.0, **_kwargs):
    import torch
    import torch.nn.functional as functional
    from transformers.models.qwen2_vl import modeling_qwen2_vl as modeling

    key_states = modeling.repeat_kv(key, module.num_key_value_groups)
    value_states = modeling.repeat_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query.float(), key_states.float().transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    weights = functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = functional.dropout(weights, p=dropout, training=module.training)
    output = torch.matmul(weights, value_states)
    return output.transpose(1, 2).contiguous(), weights


def install_official_text_scores() -> dict:
    """Patch the official eager callback before model construction/generation."""
    import torch
    from transformers.models.qwen2_vl import modeling_qwen2_vl as modeling

    current = modeling.eager_attention_forward
    if getattr(current, "_h300_fp16_text_scores", False):
        raise RuntimeError("Qwen FP16 text score compatibility already installed")
    source_path = Path(modeling.__file__).resolve()

    def stable_text_scores(module, query, key, value, attention_mask,
                           scaling, dropout=0.0, **kwargs):
        if isinstance(module, modeling.Qwen2VLAttention) and query.dtype == torch.float16:
            return fp32_score_eager_attention(
                module, query, key, value, attention_mask, scaling, dropout, **kwargs
            )
        return current(module, query, key, value, attention_mask,
                       scaling, dropout, **kwargs)

    stable_text_scores._h300_fp16_text_scores = True
    modeling.eager_attention_forward = stable_text_scores
    return {
        "scope": "official Qwen2VLAttention FP16 eager text QK scores only",
        "score_dtype": "float32", "softmax_output_dtype": "float16",
        "value_matmul_dtype": "float16", "source_path": str(source_path),
        "source_sha256": sha256(source_path),
    }


def patch_ipcv_text_scores(path: Path) -> dict:
    """Patch the hash-verified materialized author source before its first import."""
    before_hash = sha256(path)
    original = path.read_text(encoding="utf-8")
    start = original.index("class Qwen2VLAttention(nn.Module):")
    end = original.index("class Qwen2VLFlashAttention2(Qwen2VLAttention):", start)
    block = original[start:end]
    old = "torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)"
    new = "torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) / math.sqrt(self.head_dim)"
    if block.count(old) != 1 or block.count(new) != 0:
        raise RuntimeError("unexpected IPCV text score contract")
    patched = original[:start] + block.replace(old, new, 1) + original[end:]
    path.write_text(patched, encoding="utf-8")
    return {
        "scope": "IPCV Qwen2VLAttention FP16 eager text QK scores only",
        "score_dtype": "float32", "changed_score_statements": 1,
        "source_path": str(path.resolve()), "source_sha256": before_hash,
        "patched_sha256": sha256(path),
    }
