# Publication copy: comments/line endings only; algorithm AST unchanged.
# Project migration copy; retain upstream method and host notices. No experiment executed.
# Component licensing and exact source hashes: sources/FILE_DISPOSITIONS.json.
"""Frozen five-method ordinary LLaVA FP32/A800 matrix."""
from __future__ import annotations

from dataset_registry import DATASETS, DATASET_SIZES

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MODEL_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
JOB_SPECS = {
    0: ("ours", "float32", "A800"),
    1: ("pitome", "float32", "A800"),
    2: ("dtome", "float32", "A800"),
    3: ("illava", "float32", "A800"),
    4: ("ipcv", "float32", "A800"),
}
EXPERIMENT = "H303"
RELEASE = "llava15_five_method_fp32_a800_r7_fidelity_r5"
TARGET_GPU = "A800"
TARGET_DTYPE = "float32"
METHOD_LABELS = {
    "ours": "Ours (H297 optimized managed MLP+Q)",
    "pitome": "PiToMe",
    "dtome": "DToMe vision branch (DyMU visual component; no VTU)",
    "illava": "iLLaVA",
    "ipcv": "IPCV",
}
METHOD_POINTS = {
    "ours": {"source": "H297 embedded optimized CLIP", "targets_per_group": 10,
             "group_side": 8, "mlp_threshold": 0.60, "q_threshold": 0.70,
             "mlp_start_layer": 3, "q_start_layer": 5,
             "eligibility_mode": "current", "cross_layer_threshold": 0.0},
    "pitome": {"ratio": 0.96, "source": "H297 frozen author CLIP"},
    "dtome": {"source": "H297 frozen 24 CLIP thresholds; visual branch only"},
    "illava": {"layers": [12, 13, 14, 15], "reduction": 0.16},
    "ipcv": {"prune_layer": 3, "keep_ratio": 0.85, "as_layers": 7, "top_k": 10},
}
PROTOCOL = {
    "quality": "full_split_online_visual_natural_eos",
    "timing": "smolvlm_answer_blind_timing280",
    "timing_requests_by_task": {key: (10 if key.startswith("CVBench") else 20) for key in DATASETS},
    "formal_timing_repetitions": 1,
    "timing_boundary": "one-token generate; synchronized complete vision-tower wall and CUDA event",
    "primary_speed": "mean_dense_vision_wall_ms/mean_candidate_vision_wall_ms",
    "secondary_speed": "CUDA event and one-token TTFT",
    "same_request_paired_dense": True,
    "generation_safety_cap": 2048,
    "quality_attempts": 3,
    "batch_size": 1,
    "attention": "eager",
    "attention_scope": "canonical HF vision and language; native DToMe visual exception below",
    "native_dtome_visual_attention": "author MultiheadAttention with existing dispatch; primary Dense remains canonical HF; r_total=0 diagnostic only",
    "native_fidelity_acceptance": "source/call fidelity; repeat and canonical differences recorded descriptively",
    "tf32": False,
    "journal": "byte_exact_smolvlm_checkpoint_journal",
}

assert len(DATASETS) == 15 and sum(DATASET_SIZES.values()) == 32338
assert len(JOB_SPECS) == 5 and len(METHOD_LABELS) == 5
assert set(JOB_SPECS) == set(range(5))
assert {method for method, _, _ in JOB_SPECS.values()} == set(METHOD_LABELS)
assert sum(PROTOCOL["timing_requests_by_task"].values()) == 280
