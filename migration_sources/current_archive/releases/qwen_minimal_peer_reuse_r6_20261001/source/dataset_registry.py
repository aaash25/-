# Publication copy: comments/line endings only; algorithm AST unchanged.
# Project migration copy; retain upstream method and host notices. No experiment executed.
# Component licensing and exact source hashes: sources/FILE_DISPOSITIONS.json.
"""Frozen fifteen-task registry reused unchanged by H300.

The source TSVs can contain more than one split.  ``requests`` is the exact
official split that enters generation and scoring, not the raw TSV line count.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    family: str
    vlmeval_alias: str
    source_tsv: str
    requests: int
    selection: str
    score_protocol: str
    may_have_multiple_images: bool = False


SPECS = (
    DatasetSpec("HRBench4K", "HRBench4K", "HRBench4K", "HRBench4K.tsv", 800,
                "complete official file", "VLMEvalKit official MCQ scoring"),
    DatasetSpec("POPE", "POPE", "POPE", "POPE.tsv", 5127,
                "complete official file", "VLMEvalKit POPE Y/N scoring"),
    DatasetSpec("MMVP", "MMVP", "MMVP", "MMVP.tsv", 300,
                "complete official file", "VLMEvalKit MMVP paired scoring"),
    DatasetSpec("RealWorldQA", "RealWorldQA", "RealWorldQA", "RealWorldQA.tsv", 765,
                "complete official file", "VLMEvalKit official scoring"),
    DatasetSpec("MMStar", "MMStar", "MMStar", "MMStar.tsv", 1500,
                "complete official file", "VLMEvalKit official MCQ scoring"),
    DatasetSpec("MMMU_val", "MMMU", "MMMU_DEV_VAL", "MMMU_DEV_VAL.tsv", 900,
                "split == validation; exclude 150 dev rows",
                "VLMEvalKit MMMU validation scoring", True),
    DatasetSpec("ScienceQA_image_test", "ScienceQA", "ScienceQA_TEST", "ScienceQA_TEST.tsv", 2017,
                "complete image-test file", "VLMEvalKit official MCQ scoring"),
    DatasetSpec("MMBench_EN_dev_v1", "MMBench", "MMBench_DEV_EN", "MMBench_DEV_EN.tsv", 4329,
                "complete official circular-eval dev variants",
                "VLMEvalKit MMBench CircularEval scoring"),
    DatasetSpec("TextVQA_val", "TextVQA", "TextVQA_VAL", "TextVQA_VAL.tsv", 5000,
                "complete validation file", "VLMEvalKit TextVQA scoring"),
    DatasetSpec("OCRBench", "OCRBench", "OCRBench", "OCRBench.tsv", 1000,
                "complete official file", "VLMEvalKit OCRBench category scoring"),
    DatasetSpec("MME", "MME", "MME", "MME.tsv", 2374,
                "complete official file", "VLMEvalKit MME perception/cognition scoring"),
    DatasetSpec("ChartQA", "ChartQA", "ChartQA_TEST", "ChartQA_TEST.tsv", 2500,
                "complete test file", "VLMEvalKit ChartQA scoring"),
    DatasetSpec("CVBench2D", "CVBench", "CV-Bench-2D", "CV-Bench-2D.tsv", 1438,
                "complete author test_2d; audited correction of obsolete 1348 guard",
                "VLMEvalKit CV-Bench 2D scoring"),
    DatasetSpec("CVBench3D", "CVBench", "CV-Bench-3D", "CV-Bench-3D.tsv", 1200,
                "complete author test_3d; audited correction of obsolete 1290 guard",
                "VLMEvalKit CV-Bench 3D scoring"),
    DatasetSpec("AI2D", "AI2D", "AI2D_TEST", "AI2D_TEST.tsv", 3088,
                "complete official test file", "VLMEvalKit official MCQ scoring"),
)

BY_KEY = {spec.key: spec for spec in SPECS}
DATASETS = tuple(BY_KEY)
DATASET_SIZES = {spec.key: spec.requests for spec in SPECS}
FAMILIES = tuple(dict.fromkeys(spec.family for spec in SPECS))


def validate() -> None:
    if len(SPECS) != 15 or len(BY_KEY) != 15:
        raise ValueError("H300 requires fifteen distinct physical tasks")
    if len(FAMILIES) != 14:
        raise ValueError("H300 requires fourteen dataset families")
    if sum(DATASET_SIZES.values()) != 32338:
        raise ValueError("H300 request total changed")
    if DATASET_SIZES["CVBench2D"] != 1438 or DATASET_SIZES["CVBench3D"] != 1200:
        raise ValueError("complete CV-Bench split counts regressed")
    if {spec.source_tsv for spec in SPECS} != {
        "HRBench4K.tsv", "POPE.tsv", "MMVP.tsv", "RealWorldQA.tsv",
        "MMStar.tsv", "MMMU_DEV_VAL.tsv", "ScienceQA_TEST.tsv",
        "MMBench_DEV_EN.tsv", "TextVQA_VAL.tsv", "OCRBench.tsv",
        "MME.tsv", "ChartQA_TEST.tsv", "CV-Bench-2D.tsv",
        "CV-Bench-3D.tsv", "AI2D_TEST.tsv",
    }:
        raise ValueError("dataset asset registry changed")


validate()
