# Publication copy: comments/line endings only; algorithm AST unchanged.
# Project migration copy; retain upstream method and host notices. No experiment executed.
# Component licensing and exact source hashes: sources/FILE_DISPOSITIONS.json.
"""Frozen H300 matrix and SmolVLM-aligned publication protocol.

H300 covers all 15 physical tasks under one Qwen/VLMEvalKit prompt and natural
EOS contract.  H259 remains historical evidence, but its simplified prompts
and 256-token safety cap are not identical and therefore are not imported.
"""

from __future__ import annotations

from dataset_registry import DATASETS, DATASET_SIZES
CORE_DATASETS = ("CVBench2D", "HRBench4K", "POPE", "RealWorldQA")

REUSED_H259_CORE_CELLS: set[tuple[str, str]] = set()

JOB_SPECS = {
    0: ("ours", "float16"),
    1: ("ours", "float32"),
    2: ("ipcv", "float16"),
    3: ("ipcv", "float32"),
    4: ("illava", "float16"),
    5: ("illava", "float32"),
    6: ("dtome", "float16"),
    7: ("dtome", "float32"),
    8: ("pitome", "float16"),
    9: ("pitome", "float32"),
}

# Keep H259's named configurations, but rerun every cell because H300 uses
# official VLMEvalKit prompts, natural EOS and a 2048-token anomaly cap.
CONFIGS = {
    "ours": {
        "ours_s080_a085": None,
        "ours_s070_a075": None,
    },
    "ipcv": {
        "ipcv_keep85": None,
        "ipcv_keep70": None,
    },
    "illava": {
        "illava_r06": None,
        "illava_r15": None,
    },
    "dtome": {
        "dtome_t985_s1": {"method": "dtome", "start_layer": 1, "threshold": .985},
        "dtome_t9775_s1": {"method": "dtome", "start_layer": 1, "threshold": .9775},
        "dtome_t970_s1": {"method": "dtome", "start_layer": 1, "threshold": .970},
    },
    "pitome": {
        "pitome_r975_s0": {"method": "pitome", "start_layer": 0, "ratio": .975},
        "pitome_r955_s0": {"method": "pitome", "start_layer": 0, "ratio": .955},
        "pitome_r930_s0": {"method": "pitome", "start_layer": 0, "ratio": .930},
    },
}

H259_MODE = {
    "ours": {
        "ours_s080_a085": "indexed_s080_a085_fullq",
        "ours_s070_a075": "indexed_s070_a075_fullq",
    },
    "ipcv": {
        "ipcv_keep85": "ipcv_keep85",
        "ipcv_keep70": "ipcv_keep70",
    },
    "illava": {
        "illava_r06": "illava_r06",
        "illava_r15": "illava_r15",
    },
}

PROTOCOL = {
    "quality": "full_split_natural_eos",
    "timing": "dataset_matched_full_split_per_unique_image",
    "speedup": "mean_dense_cuda_ms/mean_candidate_cuda_ms",
    "timing_boundary": "complete_visual_tower_cuda_event",
    "paired_same_process_dense": True,
    # SmolVLM's formal visual-only protocol records one post-warmup
    # observation per mode and alternates the order across images.  The
    # bounded smoke phase may repeat the same path for a stability gate, but
    # those diagnostic repeats never enter the formal latency estimator.
    "formal_timing_repetitions": 1,
    "smoke_timing_repetitions": 2,
    "generation_safety_cap": 2048,
    "attention_backend": "eager",
    "quality_execution": "online_visual_per_question",
    "generation_attempts": 3,
    "journal": "byte_exact_smolvlm_checkpoint_journal",
}


def modes(method: str) -> tuple[str, ...]:
    return ("dense", *CONFIGS[method])


def validate() -> None:
    expected_matrix = {
        ("ours", "float16"),
        ("ours", "float32"),
        ("ipcv", "float16"),
        ("ipcv", "float32"),
        ("illava", "float16"),
        ("illava", "float32"),
        ("dtome", "float16"),
        ("dtome", "float32"),
        ("pitome", "float16"),
        ("pitome", "float32"),
    }
    if set(JOB_SPECS) != set(range(10)) or set(JOB_SPECS.values()) != expected_matrix:
        raise ValueError("H300 full matrix changed")
    if set(CONFIGS) != {"ours", "ipcv", "illava", "dtome", "pitome"}:
        raise ValueError("strict peer set changed")
    if len(DATASETS) != 15 or sum(DATASET_SIZES.values()) != 32338:
        raise ValueError("15-task full-dataset scope changed")
    if any(not values for values in CONFIGS.values()):
        raise ValueError("method without a frozen candidate")


validate()
