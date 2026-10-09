# Publication copy: comments/line endings only; algorithm AST unchanged.
# Project migration copy; retain upstream method and host notices. No experiment executed.
# Component licensing and exact source hashes: sources/FILE_DISPOSITIONS.json.
"""Reuse H125's actual SmolVLM adapter without running its install/main routine.

This is a migration-validation bridge, not a new low-overhead implementation.
H125's diagnostic work is retained and must not be presented as a polished
deployment baseline without the planned hot-path review. The bridge overrides
only destination aggregation: independent FP32 segment sums avoid cancellation
between unrelated destinations in H125's global prefix-subtraction variant.
The release-local H125 copy also supports deferred reporting for timing; the
bridge defaults to full diagnostics for migration, calibration and quality.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parents[3]
SOURCE = ROOT / "route/hypotheses/v439_h125_dtome_smolvlm_pope_full300/evaluate.py"


def load_h125():
    spec = importlib.util.spec_from_file_location("peer_migration_h125", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_merge_by_destination(self, tensor, src_b, src_s, dst_idx):
    """H125 layout with independent, stable-ordered FP32 destination sums.

    Matching and compact-token order are unchanged. This is a numerical
    compatibility variant, not a claim of bitwise author scatter ordering.
    """
    torch = self.torch
    src = tensor[..., ::2, :]
    dst = tensor[..., 1::2, :].clone()
    if src_b.numel() == 0:
        return torch.cat((src, dst), dim=1)
    flat_dst = dst.reshape(-1, dst.shape[-1])
    flat_indices = src_b * dst.shape[1] + dst_idx
    order = torch.argsort(flat_indices, stable=True)
    sorted_indices = flat_indices.index_select(0, order)
    sorted_tokens = src[src_b, src_s, :].index_select(0, order)
    unique_indices, lengths = torch.unique_consecutive(sorted_indices, return_counts=True)
    # Restart accumulation at every destination; never subtract global prefixes.
    segment_sums = torch.segment_reduce(sorted_tokens.float(), 'sum', lengths=lengths)
    flat_dst[unique_indices] = (
        flat_dst.index_select(0, unique_indices).float() + segment_sums
    ).to(flat_dst.dtype)
    merged = torch.cat((src, flat_dst.reshape_as(dst)), dim=1)
    merged[src_b, src_s, :] = 0
    return merged


class DToMeLegacyBridge:
    def __init__(self, encoder, thresholds, *, diagnostic_trace=True):
        self.encoder, self.thresholds, self.adapter = encoder, list(thresholds), None
        self.diagnostic_trace = diagnostic_trace

    def __enter__(self):
        if self.encoder.training or self.encoder.config._attn_implementation != "eager":
            raise RuntimeError("Migration contract requires eval and eager")
        if getattr(self.encoder, "_peer_migration_owner", None) is not None:
            raise RuntimeError("An encoder port is already active")
        module = load_h125()
        model = SimpleNamespace(model=SimpleNamespace(vision_model=SimpleNamespace(encoder=self.encoder)))
        class StableDestinationDToMeAdapter(module.DToMeAdapter):
            _apply_merge = apply_merge_by_destination

        self.adapter = StableDestinationDToMeAdapter(
            model, self.thresholds, diagnostic_trace=self.diagnostic_trace
        )
        self.encoder._peer_migration_owner = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.adapter is not None:
            self.adapter.close()
            del self.encoder._peer_migration_owner
            self.adapter = None

    @property
    def trace(self):
        return self.adapter.layer_stats

    def summary(self):
        return self.adapter.summary()
