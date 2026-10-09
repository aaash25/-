# Publication copy: comments/line endings only; algorithm AST unchanged.
# Project migration copy; retain upstream method and host notices. No experiment executed.
# Component licensing and exact source hashes: sources/FILE_DISPOSITIONS.json.
"""Shared frozen contracts for the four-peer SmolVLM production campaign."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path


VERSION = "four-peer-formal-rerun-20260930-r1"
BUDGETS = ("B70", "B55", "B40")
BUDGET_TARGETS = {"B70": 0.70, "B55": 0.55, "B40": 0.40}
METHODS = ("dtome", "pitome", "illava_vit", "ipcv")
WORKER_METHODS = {
    "A": ("dtome", "illava_vit"),
    "B": ("dtome", "pitome", "ipcv"),
}
CALIBRATION_METHODS = {
    "A": ("dtome", "illava_vit"),
    "B": ("pitome", "ipcv"),
}
DATASET_COUNTS = {
    "HRBench4K": 800,
    "POPE": 5127,
    "MMVP": 300,
    "RealWorldQA": 765,
    "MMStar": 1500,
    "MMMU_val": 900,
    "ScienceQA_image_test": 2017,
    "MMBench_EN_dev_v1": 4329,
    "TextVQA_val": 5000,
    "OCRBench": 1000,
    "MME": 2374,
    "ChartQA": 2500,
    "CVBench2D": 1438,
    "CVBench3D": 1200,
    "AI2D": 3088,
}
DATASET_ORDERS = {
    "A": (
        "MMVP", "RealWorldQA", "HRBench4K", "MMMU_val", "OCRBench",
        "CVBench3D", "CVBench2D", "MMStar", "ScienceQA_image_test", "MME",
        "ChartQA", "AI2D", "MMBench_EN_dev_v1", "TextVQA_val", "POPE",
    ),
    "B": (
        "POPE", "TextVQA_val", "MMBench_EN_dev_v1", "AI2D", "ChartQA",
        "MME", "ScienceQA_image_test", "MMStar", "CVBench2D", "CVBench3D",
        "OCRBench", "MMMU_val", "HRBench4K", "RealWorldQA", "MMVP",
    ),
}
DTOME_PARTITION_COUNTS = {
    "AI2D": (1544, 1544), "ChartQA": (1250, 1250),
    "CVBench2D": (719, 719), "CVBench3D": (600, 600),
    "HRBench4K": (400, 400), "MMBench_EN_dev_v1": (2165, 2164),
    "MME": (1187, 1187), "MMMU_val": (450, 450),
    "MMStar": (750, 750), "MMVP": (150, 150),
    "OCRBench": (500, 500), "POPE": (2563, 2564),
    "RealWorldQA": (382, 383), "ScienceQA_image_test": (1009, 1008),
    "TextVQA_val": (2500, 2500),
}
CALIBRATION_SEED = "five-peer-calibration140-answer-blind-v1"
TIMING_SEED = "five-peer-timing280-answer-blind-v1"
AA_SEED = "five-peer-timing-aa20-answer-blind-v1"

# Values were extracted from the three official DyMU SigLIP-SO400M checkpoints
# used in H125.  The original filenames and hashes keep the provenance explicit.
DTOME_THRESHOLDS = {
    "dtome_324out": {
        "filename": "siglip-so400m-patch14-384-tome-324out.pth",
        "sha256": "01412a9a98129794f1f9cef814e3c07ecc266c969907de1ee168441a2221e5d6",
        "thresholds": [
            1.0, .94140625, .99609375, .97265625, .94140625, .94140625,
            .94140625, .94140625, .94140625, .94140625, .94140625,
            .94140625, .94140625, .94140625, .94140625, .94140625,
            .94140625, .94140625, .94140625, .94140625, .94140625,
            .94140625, .94140625, .94140625, .94140625, .94140625,
            .94140625,
        ],
    },
    "dtome_162out": {
        "filename": "siglip-so400m-patch14-384-tome-162out.pth",
        "sha256": "6b967d0780ad72ed43d81df9af9bac424e149d3ff1aea98dd485f0d1f35aed11",
        "thresholds": [
            1.0, .99609375, .94140625, .94140625, .94140625, .94140625,
            .91015625, .94140625, .94140625, .94140625, .91015625,
            .94140625, .91015625, .94140625, .91015625, .91015625,
            .94140625, .91015625, .91015625, .91015625, .91015625,
            .91015625, .87890625, .87890625, .87890625, .87890625,
            .87890625,
        ],
    },
    "dtome_81out": {
        "filename": "siglip-so400m-patch14-384-tome-81out.pth",
        "sha256": "a7cf7338ff042295f6518da167d7af994aabe73539774d4e766847f616b97754",
        "thresholds": [
            1.0, .99609375, .94140625, .94140625, .91015625, .94140625,
            .91015625, .94140625, .94140625, .91015625, .91015625,
            .91015625, .91015625, .91015625, .91015625, .91015625,
            .91015625, .91015625, .91015625, .87890625, .87890625,
            .87890625, .87890625, .84765625, .84765625, .81640625,
            .81640625,
        ],
    },
}
DTOME_PROVENANCE = {
    "repo": "mikewang/DyMU",
    "revision": "1ba4427ac90ca28eff5904fbc419c2381b93a067",
}


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("wb") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2,
                                allow_nan=False).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def validate_manifests(manifests: dict[str, dict]) -> None:
    if set(manifests) != set(DATASET_COUNTS):
        raise ValueError("The production campaign requires all 15 physical manifests")
    for name, expected in DATASET_COUNTS.items():
        manifest = manifests[name]
        if manifest.get("dataset") != name or len(manifest.get("rows", ())) != expected:
            raise ValueError(f"{name}: expected the frozen complete {expected}-request manifest")


def _rank(seed: str, dataset: str, request_id: str) -> tuple[str, str]:
    payload = f"{seed}|{dataset}|{request_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), str(request_id)


def select_by_family(manifests: dict[str, dict], *, per_family: int, seed: str) -> list[tuple[str, dict]]:
    """Return an answer-blind equal-family selection (CV-Bench is one family)."""
    if per_family < 2 or per_family % 2:
        raise ValueError("per_family must be a positive even number")
    selected: list[tuple[str, dict]] = []
    for name in DATASET_COUNTS:
        count = per_family // 2 if name in ("CVBench2D", "CVBench3D") else per_family
        ranked = sorted(manifests[name]["rows"],
                        key=lambda row: _rank(seed, name, str(row["id"])))
        selected.extend((name, row) for row in ranked[:count])
    expected = 13 * per_family + per_family
    if len(selected) != expected:
        raise AssertionError((len(selected), expected))
    return selected


def dtome_rows(rows: list[dict], dataset: str, worker: str) -> list[dict]:
    if worker not in ("A", "B"):
        raise ValueError(worker)
    ordered = sorted(rows, key=lambda row: (
        hashlib.sha256(f"{dataset}\0{row['id']}".encode("utf-8")).hexdigest(),
        str(row["id"]),
    ))
    count_a, count_b = DTOME_PARTITION_COUNTS[dataset]
    if len(ordered) != count_a + count_b:
        raise ValueError(f"{dataset}: DToMe partition count does not match manifest")
    return ordered[:count_a] if worker == "A" else ordered[count_a:]


def dtome_timing_rows(rows: list[dict], dataset: str, worker: str) -> list[dict]:
    """Split an even-sized timing subset exactly in half without labels."""
    if worker not in ("A", "B") or len(rows) % 2:
        raise ValueError("DToMe timing split requires worker A/B and an even row count")
    ordered = sorted(rows, key=lambda row: (
        hashlib.sha256(f"{dataset}\0{row['request_id']}".encode("utf-8")).hexdigest(),
        str(row["request_id"]),
    ))
    middle = len(ordered) // 2
    return ordered[:middle] if worker == "A" else ordered[middle:]


def candidate_configs(method: str) -> list[dict]:
    if method == "dtome":
        return [{"checkpoint": name} for name in DTOME_THRESHOLDS]
    if method == "pitome":
        return [{"ratio": value} for value in
                (.995, .990, .985, .980, .975, .970, .965, .960, .955, .950, .940, .930)]
    if method == "illava_vit":
        return [{"zero_based_layers": [5, 6, 7, 8], "removed_per_layer": value}
                for value in (32, 48, 64, 80, 92, 104, 120, 136, 152, 160)]
    if method == "ipcv":
        return [{"zero_based_prune_layer": 3, "keep_ratio": value,
                 "as_layers": 7, "top_k": 10}
                for value in (.95, .85, .75, .65, .55, .45, .35, .25, .15)]
    if method == "ficoco_v":
        return [{"start_layer_zero_based": 11, "removed_per_layer": value,
                 "lambda": .35, "epsilon": .998, "local_penalty": 2.0}
                for value in (8, 12, 16, 20, 24, 28, 32, 36, 40, 44)]
    raise ValueError(method)


def config_id(method: str, config: dict) -> str:
    return f"{method}-{digest(config)[:12]}"


def method_context(encoder, method: str, config: dict, *, disabled: bool = False,
                   diagnostic_trace: bool = True):
    """DToMe may defer diagnostics for timing; other peer paths are unchanged."""
    if method == "dense":
        return contextlib.nullcontext(None)
    if method == "dtome":
        from dtome_legacy_bridge import DToMeLegacyBridge
        entry = DTOME_THRESHOLDS[config["checkpoint"]]
        thresholds = [2.0] * len(entry["thresholds"]) if disabled else entry["thresholds"]
        return DToMeLegacyBridge(encoder, thresholds, diagnostic_trace=diagnostic_trace)
    if method in ("pitome", "illava_vit"):
        from smol_merge_ports import SmolMergePort
        if method == "pitome":
            return SmolMergePort(encoder, method, ratio=1.0 if disabled else config["ratio"])
        return SmolMergePort(
            encoder, method,
            merge_layers=tuple(config["zero_based_layers"]),
            removed=0 if disabled else config["removed_per_layer"],
        )
    if method == "ipcv":
        from smol_ipcv_port import SmolIPCVPort
        return SmolIPCVPort(
            encoder, prune_layer=config["zero_based_prune_layer"],
            keep_ratio=1.0 if disabled else config["keep_ratio"],
            as_layers=config["as_layers"], top_k=config["top_k"],
        )
    if method == "ficoco_v":
        from smol_ficoco_port import SmolFiCoCoPort
        return SmolFiCoCoPort(
            encoder, start_layer=config["start_layer_zero_based"],
            removed=0 if disabled else config["removed_per_layer"],
            epsilon=config["epsilon"], lam=config["lambda"],
            penalty=config["local_penalty"],
        )
    raise ValueError(method)


def compression_record(method: str, port, *, include_trace: bool) -> dict:
    if method == "dense":
        return {"active": False, "original_tokens": 729}
    if method == "dtome":
        full = port.summary()
        record = {
            "active": full["total_merged_edges"] > 0,
            "original_tokens": full["restored_tokens_per_crop"],
            "final_padded_tokens": full["final_padded_tokens"],
            "final_valid_tokens_per_crop": full["final_valid_tokens_per_crop"],
            "total_merged_edges": full["total_merged_edges"],
            "membership_exact": full["membership_exact"],
        }
        trace = full["layers"]
    else:
        trace = port.trace
        attention = [int(row["attention_tokens"]) for row in trace]
        mlp = [int(row["mlp_tokens"]) for row in trace]
        output = [int(row.get("output_tokens", row["mlp_tokens"])) for row in trace]
        record = {
            "active": min(attention + mlp + output) < attention[0],
            "original_tokens": attention[0],
            "min_attention_tokens": min(attention),
            "min_mlp_tokens": min(mlp),
            "final_tokens": output[-1],
        }
    if include_trace:
        record["trace"] = trace
    return record


def flops_ratio(method: str, compression: dict, hidden_size: int,
                intermediate_size: int, layers: int) -> float:
    """Analytic dense-relative transformer-block FLOPs from the realized trace."""
    def layer_cost(attention_tokens: int, mlp_tokens: int) -> int:
        n, m, d, f = attention_tokens, mlp_tokens, hidden_size, intermediate_size
        return 4 * n * d * d + 2 * n * n * d + 2 * m * d * f

    dense = layers * layer_cost(729, 729)
    trace = compression.get("trace")
    if not trace or len(trace) != layers:
        raise ValueError("A complete layer trace is required for FLOPs calibration")
    if method == "dtome":
        outputs = [int(row["padded_tokens"]) for row in trace]
        attention = [729] + outputs[:-1]
        mlp = outputs
    else:
        attention = [int(row["attention_tokens"]) for row in trace]
        mlp = [int(row["mlp_tokens"]) for row in trace]
    ratio = sum(layer_cost(n, m) for n, m in zip(attention, mlp)) / dense
    if not 0 < ratio <= 1.05 or not math.isfinite(ratio):
        raise ValueError(f"Invalid realized FLOPs ratio: {ratio}")
    return ratio


def add_self_digest(value: dict) -> dict:
    result = dict(value)
    result["content_digest"] = digest(result)
    return result


def verify_self_digest(value: dict) -> dict:
    result = dict(value)
    observed = result.pop("content_digest", None)
    if observed != digest(result):
        raise ValueError("Frozen JSON content digest mismatch")
    result["content_digest"] = observed
    return result
