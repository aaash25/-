# Publication copy: comments/line endings only; algorithm AST unchanged.
# Project migration copy; retain upstream method and host notices. No experiment executed.
# Component licensing and exact source hashes: sources/FILE_DISPOSITIONS.json.
"""Resumable five-method LLaVA quality and SmolVLM timing280 job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
import uuid
from pathlib import Path

import grid
from launch_plan import SELECTED_JOBS
from prepare_timing import selected as selected_timing_rows
from checkpoint_journal import Journal
from llava_runtime import frozen_source_hashes, load
from release_integrity import verify_release
from fidelity_policy import assess_repeat
from prepare_manifests import atomic_json, digest_json

HERE = Path(__file__).resolve().parent
EXPERIMENT = grid.EXPERIMENT
RELEASE = grid.RELEASE
AA_SEED = "five-peer-timing-aa20-answer-blind-v1"
MODES = ("dense", "candidate")


def verified_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    saved = value.pop("content_digest")
    if digest_json(value) != saved:
        raise RuntimeError(f"content digest mismatch: {path}")
    value["content_digest"] = saved
    return value


def load_manifests(directory: Path) -> dict[str, dict]:
    manifests = {dataset: verified_json(directory / f"{dataset}.json")
                 for dataset in grid.DATASETS}
    for dataset, manifest in manifests.items():
        ids = [str(row["id"]) for row in manifest["rows"]]
        if (manifest["dataset"] != dataset or not manifest["full_selected_split"]
                or len(ids) != grid.DATASET_SIZES[dataset]
                or len(ids) != len(set(ids)) or len(manifest["metadata"]) != len(ids)):
            raise RuntimeError(f"full-split LLaVA manifest changed: {dataset}")
    return manifests


def select_smoke(manifests: dict[str, dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    selected = {}
    for dataset in ("HRBench4K", "MMMU_val"):
        rows = manifests[dataset]["rows"]
        if dataset == "MMMU_val":
            matches = [row for row in rows if len(row["images"]) > 1]
            if not matches:
                raise RuntimeError("smoke requires one real multi-image MMMU request")
            selected[dataset] = matches[:1]
        else:
            selected[dataset] = rows[:1]
    timing = [dict(dataset=dataset, request_id=str(row["id"]), **{
        key: row[key] for key in ("messages", "images", "image_sha256", "prompt_sha256")
    }) for dataset, rows in selected.items() for row in rows]
    return timing, selected


def selected_work(manifests: dict[str, dict], timing: dict, phase: str):
    if phase == "smoke":
        return select_smoke(manifests)
    if timing.get("stage") != "answer_blind_timing280" or len(timing["rows"]) != 280:
        raise RuntimeError("formal timing280 scope changed")
    if timing.get("contains_labels_or_predictions") is not False:
        raise RuntimeError("timing selection is not answer-blind")
    if timing["manifest_digests"] != {
        key: value["content_digest"] for key, value in manifests.items()
    }:
        raise RuntimeError("timing/quality manifests differ")
    if timing["rows"] != selected_timing_rows(manifests):
        raise RuntimeError("timing280 request selection differs from SmolVLM rule")
    by_dataset = {key: [] for key in grid.DATASETS}
    for row in timing["rows"]:
        by_dataset[row["dataset"]].append(row)
    if {key: len(value) for key, value in by_dataset.items()} != grid.PROTOCOL["timing_requests_by_task"]:
        raise RuntimeError("per-task timing280 scope changed")
    return timing["rows"], {key: value["rows"] for key, value in manifests.items()}


def aa_ids(timing_rows: list[dict]) -> set[tuple[str, str]]:
    if len(timing_rows) < 20:
        return set()
    ordered = sorted(timing_rows, key=lambda row: (
        hashlib.sha256((f"{AA_SEED}|{row['dataset']}|{row['request_id']}").encode()).hexdigest(),
        f"{row['dataset']}|{row['request_id']}",
    ))
    return {(row["dataset"], str(row["request_id"])) for row in ordered[:20]}


def source_hashes() -> dict[str, str]:
    names = ("checkpoint_journal.py", "dataset_registry.py", "grid.py", "prompting.py",
             "llava_runtime.py", "prepare_manifests.py", "prepare_timing.py",
             "run_job.py", "audit_smoke.py", "audit.py",
             "score_predictions.py", "score_all.py", "run_on_idle_gpu.sh",
             "launch_contract.py", "campaign_preflight.py")
    names += ("release_integrity.py", "check_environment.py", "environment_lock.json",
              "fidelity_policy.py", "FIDELITY_EVIDENCE.json", "launch_plan.py")
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names}


def ensure_contract(output: Path, value: dict) -> None:
    path = output / "CONTRACT.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError("job contract changed; use a new result directory")
    else:
        if any(output.iterdir()):
            raise RuntimeError("uncontracted existing result directory")
        atomic_json(path, value)


def status(output: Path, state: str, current: dict | None,
           committed: int, expected: int, started: float) -> None:
    atomic_json(output / "STATUS.json", {
        "experiment": EXPERIMENT, "release": RELEASE, "state": state,
        "current": current, "committed_units": committed,
        "expected_units": expected, "missing_units": expected - committed,
        "elapsed_seconds_this_epoch": time.monotonic() - started,
        "updated_unix": time.time(),
    })


def retry_three(operation):
    for attempt in range(1, 4):
        try:
            return operation(), attempt
        except Exception:
            if attempt == 3:
                raise
    raise AssertionError("unreachable")


def export_quality(path: Path, manifest: dict, journals: dict[str, Journal]) -> int:
    rows = []
    for row in manifest["rows"]:
        key = str(row["id"])
        if all(journals[mode].contains(key) for mode in MODES):
            rows.append({
                "dataset": manifest["dataset"], "id": key,
                "image_key": row["image_key"],
                "prompt_sha256": row["prompt_sha256"],
                "execution": "online_visual_per_question",
                "modes": {mode: journals[mode].result(key)["output"] for mode in MODES},
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".pending")
    with temp.open("wb") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return len(rows)


def run(args) -> dict:
    import torch

    release_sha256 = verify_release(HERE)
    if args.job not in SELECTED_JOBS:
        raise ValueError(args.job)
    if args.max_seconds < 600:
        raise ValueError("GPU budget must be at least 600 seconds")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    manifests = load_manifests(args.manifest_dir)
    timing = verified_json(args.timing_manifest)
    timing_rows, quality_rows = selected_work(manifests, timing, args.phase)
    method, dtype, target_gpu = grid.JOB_SPECS[args.job]
    output = args.output_root / f"job_{args.job}"
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "experiment": EXPERIMENT, "release": RELEASE, "phase": args.phase,
        "job": args.job, "method": method, "method_label": grid.METHOD_LABELS[method],
        "dtype": dtype, "target_gpu": target_gpu,
        "model": grid.MODEL_ID, "revision": grid.MODEL_REVISION,
        "protocol": grid.PROTOCOL, "method_point": grid.METHOD_POINTS[method],
        "manifests": {key: value["content_digest"] for key, value in manifests.items()},
        "timing_digest": timing["content_digest"],
        "data_root": str(args.data_root.resolve()),
        "sources": source_hashes(), "frozen_h297_sources": frozen_source_hashes(),
        "release_hashes_sha256": release_sha256,
    }
    ensure_contract(output, contract)
    runtime = load(args.job, args.snapshot, output / "scratch", phase=args.phase)
    atomic_json(output / "RUNTIME.json", runtime.receipt)
    torch.set_grad_enabled(False)
    expected = len(timing_rows) + 2 * sum(len(rows) for rows in quality_rows.values())
    started = time.monotonic()
    deadline = started + args.max_seconds - 30
    journal_root = output / "journals"
    committed = sum(
        path.read_bytes().count(b"\n")
        for path in journal_root.rglob("segment_*.jsonl")
    ) if journal_root.is_dir() else 0
    current = None
    epoch = uuid.uuid4().hex
    paused = False
    warmed = set()
    aa = aa_ids(timing_rows)
    status(output, "starting", current, committed, expected, started)
    try:
        with Journal(output / "journals" / "timing", contract | {"kind": "timing"},
                     segment_records=256) as timing_journal:
            timing_journal.missing([f"{row['dataset']}/{row['request_id']}" for row in timing_rows])
            for index, row in enumerate(timing_rows):
                key = f"{row['dataset']}/{row['request_id']}"
                if timing_journal.contains(key):
                    continue
                if time.monotonic() >= deadline:
                    paused = True
                    break
                current = {"stage": "timing", "key": key}
                prepared = {mode: runtime.prepare(row, args.data_root, mode) for mode in MODES}
                for mode in MODES:
                    if mode not in warmed:
                        runtime.generate(prepared[mode], mode, max_new_tokens=1)
                        warmed.add(mode)
                order = MODES if index % 2 == 0 else tuple(reversed(MODES))
                observations = {}
                first_features = {}
                for mode in order:
                    feature, measures, profile = runtime.timed_visual(prepared[mode], mode)
                    expected_tokens = 232 if method == "pitome" and mode == "candidate" else 576
                    expected_shape = (len(row["images"]), expected_tokens, runtime.dense_tower.config.hidden_size)
                    finite = bool(torch.isfinite(feature).all().item())
                    if tuple(feature.shape) != expected_shape or not finite:
                        raise RuntimeError(f"invalid visual feature boundary: {key}/{mode}")
                    observations[mode] = {**measures, "profile": profile,
                                          "feature_shape": list(feature.shape), "features_finite": finite}
                    if args.phase == "smoke":
                        first_features[mode] = feature
                    else:
                        del feature
                smoke_repeat = None
                if args.phase == "smoke":
                    smoke_repeat = {}
                    for mode in reversed(order):
                        feature, measures, profile = runtime.timed_visual(prepared[mode], mode)
                        first = first_features[mode]
                        shape_equal = first.shape == feature.shape
                        finite = bool(torch.isfinite(first).all().item() and torch.isfinite(feature).all().item())
                        atol = 2e-3 if dtype == "float16" else 5e-4
                        stable = bool(shape_equal and torch.allclose(first, feature, atol=atol, rtol=1e-5))
                        same_profile = profile == observations[mode]["profile"]
                        delta = first.float() - feature.float() if shape_equal and finite else None
                        smoke_repeat[mode] = {**measures, "feature_close": stable,
                            "profile_equal": same_profile, "profile": profile,
                            "shape_equal": shape_equal, "features_finite": finite,
                            "rtol": 1e-5, "atol": atol,
                            "max_abs": float(delta.abs().max().item()) if delta is not None else None,
                            "relative_l2": float(delta.norm().item() / max(first.float().norm().item(), 1e-12)) if delta is not None else None,
                            "relative_l2_reference": "first observation"}
                        admission = assess_repeat(method, mode, smoke_repeat[mode])
                        smoke_repeat[mode]["admission"] = admission
                        if not admission["accepted"]:
                            raise RuntimeError(f"invalid smoke repeat: {key}/{mode}")
                        del feature
                    first_features.clear()
                aa_pair = None
                if (row["dataset"], str(row["request_id"])) in aa:
                    aa_pair = [runtime.timed_visual(prepared["dense"], "dense")[1]
                               for _ in range(2)]
                timing_journal.commit(key, {
                    "dataset": row["dataset"], "request_id": str(row["request_id"]),
                    "method": method, "dtype": dtype, "epoch": epoch,
                    "order": list(order), "observations": observations,
                    "smoke_repeat": smoke_repeat,
                    "dense_aa": aa_pair,
                    "timing_protocol": "smolvlm_formal_one_post_warmup_observation",
                })
                committed += 1
                if committed % 25 == 0:
                    status(output, "running", current, committed, expected, started)

        if not paused:
            for dataset, rows in quality_rows.items():
                manifest = manifests[dataset]
                with Journal(output / "journals" / "quality" / "dense" / dataset,
                             contract | {"kind": "quality", "dataset": dataset, "mode": "dense"},
                             segment_records=256) as dense_journal, Journal(
                             output / "journals" / "quality" / "candidate" / dataset,
                             contract | {"kind": "quality", "dataset": dataset, "mode": "candidate"},
                             segment_records=256) as candidate_journal:
                    journals = {"dense": dense_journal, "candidate": candidate_journal}
                    for journal in journals.values():
                        journal.missing([str(row["id"]) for row in rows])
                    for index, row in enumerate(rows):
                        key = str(row["id"])
                        order = MODES if index % 2 == 0 else tuple(reversed(MODES))
                        for mode in order:
                            if journals[mode].contains(key):
                                continue
                            if time.monotonic() >= deadline:
                                paused = True
                                break
                            current = {"stage": "quality", "dataset": dataset,
                                       "request_id": key, "mode": mode}
                            try:
                                result, attempts = retry_three(lambda: runtime.generate(
                                    runtime.prepare(row, args.data_root, mode), mode,
                                    max_new_tokens=grid.PROTOCOL["generation_safety_cap"],
                                ))
                            except Exception:
                                atomic_json(output / "attempt_failures" / f"{uuid.uuid4().hex}.json",
                                            current | {"epoch": epoch, "traceback": traceback.format_exc()})
                                raise
                            journals[mode].commit(key, {
                                "dataset": dataset, "request_id": key, "mode": mode,
                                "epoch": epoch, "attempts": attempts,
                                "execution": "online_visual_per_question",
                                "prompt_sha256": row["prompt_sha256"], "output": result,
                            })
                            committed += 1
                            if committed % 25 == 0:
                                status(output, "running", current, committed, expected, started)
                        if paused:
                            break
                    if args.phase == "formal":
                        export_quality(output / "quality" / f"{dataset}.quality.jsonl",
                                       manifest, journals)
                if paused:
                    break
        # Recount from journals on resume; prior committed records are not
        # regenerated and cannot be inferred from this epoch's counter.
        committed_total = sum(
            path.read_bytes().count(b"\n")
            for path in journal_root.rglob("segment_*.jsonl")
        )
        state = "paused_budget" if paused else "complete"
        if not paused and committed_total != expected:
            raise RuntimeError(f"incomplete job coverage: {committed_total}/{expected}")
        final = {"state": state, "experiment": EXPERIMENT, "release": RELEASE,
                 "phase": args.phase, "job": args.job, "method": method, "dtype": dtype,
                 "committed_units": committed_total, "expected_units": expected,
                 "wall_seconds": time.monotonic() - started,
                 "max_gpu_reserved_bytes": torch.cuda.max_memory_reserved()}
        atomic_json(output / "FINAL_STATUS.json", final)
        status(output, state, current, committed_total, expected, started)
        return final
    except BaseException:
        atomic_json(output / "FINAL_STATUS.json", {
            "state": "failed", "job": args.job, "method": method, "dtype": dtype,
            "current": current, "traceback": traceback.format_exc(),
        })
        status(output, "failed", current, committed, expected, started)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--job", type=int, choices=SELECTED_JOBS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--timing-manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=259200)
    arguments = parser.parse_args()
    result = run(arguments)
    raise SystemExit(0 if result["state"] == "complete" else 75)
