# Publication copy: comments/line endings only; algorithm AST unchanged.
# Project migration copy; retain upstream method and host notices. No experiment executed.
# Component licensing and exact source hashes: sources/FILE_DISPOSITIONS.json.
"""Run one H300 Qwen job with the mature SmolVLM experiment semantics."""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import os
import statistics
import time
import traceback
import uuid
from pathlib import Path

import grid
import method_runtime
from checkpoint_journal import Journal, canonical, sync_directory
from prepare_manifests import atomic_json, digest_json


EXPERIMENT = "H300"
RELEASE = "smolvlm_aligned_eager_r1"
HERE = Path(__file__).resolve().parent
MODEL_FILES = {
    "config.json": (1196, "422adefa19e62dd175961cec85bc0400344fe5bf9b22bd1182e05aaae78556e0"),
    "model.safetensors.index.json": (56411, "260ab9fa1418d6d6ab79daa1d9da2c47264f3b72edb4630fc799077ac67d27c6"),
    "model-00001-of-00002.safetensors": (3988609112, None),
    "model-00002-of-00002.safetensors": (429441656, None),
    "preprocessor_config.json": (347, "b5eaad0c2815f07631535dcc58f3c462b0d73693638ad21d19f3c50820eae1cc"),
    "tokenizer.json": (7029741, "cb63a0a23eef3d5b01063a9880a1925a65aaf4d1591d519910ee3527852950a0"),
    "tokenizer_config.json": (4190, "ff5c4fd898fe8c39591eb70e5d39d2782802d4204d6ae9ba1223252f354842a0"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files() -> dict[str, str]:
    paths = (
        HERE / "grid.py", HERE / "dataset_registry.py",
        HERE / "method_runtime.py", HERE / "h259_bridge.py",
        HERE / "fp16_eager_stability.py",
        HERE / "illava_native_adapter.py", HERE / "illava_native_source.py",
        HERE / "references" / "illava_qwen2vl_d2d2918.py", HERE / "run_job.py",
        HERE / "checkpoint_journal.py",
        HERE / "qwen_peer_ports.py", HERE / "peer_port_core.py",
        HERE / "references" / "ours_evaluate.py",
        HERE / "references" / "ipcv_evaluate.py",
        HERE / "references" / "illava_evaluate.py",
    )
    return {path.name: sha256(path) for path in paths}


def verify_snapshot(snapshot: Path) -> dict:
    snapshot = snapshot.resolve()
    checks = {}
    for name, (size, expected_hash) in MODEL_FILES.items():
        path = snapshot / name
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"Qwen snapshot file missing/size mismatch: {path}")
        observed = sha256(path) if expected_hash else None
        if expected_hash and observed != expected_hash:
            raise RuntimeError(f"Qwen snapshot hash mismatch: {path}")
        checks[name] = {"bytes": size, "sha256": observed}
    return {
        "snapshot": str(snapshot),
        "revision": method_runtime.MODEL_REVISION,
        "files": checks,
    }


def load_manifest(path: Path, phase: str = "formal") -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    digest = value.pop("content_digest")
    if digest_json(value) != digest:
        raise RuntimeError(f"manifest digest mismatch: {path}")
    value["content_digest"] = digest
    expected_full = grid.DATASET_SIZES[value["dataset"]]
    smoke_fixture = (
        phase == "smoke"
        and value.get("fixture_protocol") == "h300-kaggle-t4x2-eager-preflight-r1"
        and not value.get("full_selected_split")
        and 0 < value.get("row_count", 0) <= 4
    )
    expected = value["row_count"] if smoke_fixture else expected_full
    if (
        (not smoke_fixture and not value["full_selected_split"])
        or value["row_count"] != expected
        or len(value["rows"]) != expected
        or len(value["metadata"]) != expected
    ):
        raise RuntimeError(f"manifest scope mismatch: {path}")
    ids = [str(row["id"]) for row in value["rows"]]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate manifest IDs: {path}")
    return value


def ensure_contract(path: Path, contract: dict) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("contract changed: use a new H300 result directory")
        return
    if any(path.parent.iterdir()):
        raise RuntimeError("uncontracted output directory")
    atomic_json(path, contract)


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def prepare_inputs(runtime, row: dict, data_root: Path) -> dict:
    from PIL import Image

    content, opened, images = [], [], []
    try:
        for item in row["messages"]:
            if item["type"] == "text":
                content.append({"type": "text", "text": item["value"]})
            else:
                path = (data_root / item["value"]).resolve()
                if not path.is_relative_to(data_root.resolve()) or not path.is_file():
                    raise RuntimeError(f"missing/outside image: {path}")
                handle = Image.open(path)
                opened.append(handle)
                image = handle.convert("RGB")
                images.append(image)
                content.append({"type": "image"})
        conversation = [{"role": "user", "content": content}]
        text = runtime.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        values = runtime.processor(
            text=[text], images=images, padding=True, return_tensors="pt"
        )
        return {
            key: value.to("cuda:0") if hasattr(value, "to") else value
            for key, value in values.items()
        }
    finally:
        for image in images:
            image.close()
        for handle in opened:
            handle.close()


def feature_comparison(first, second, dtype) -> dict:
    import torch

    shape_equal = tuple(first.shape) == tuple(second.shape)
    if not shape_equal:
        return {"shape_equal": False, "exact": False, "numeric_close": False}
    diff = first.float() - second.float()
    max_abs = float(diff.abs().max().item())
    reference_norm = float(torch.linalg.vector_norm(first.float()).item())
    relative_l2 = (
        float(torch.linalg.vector_norm(diff).item() / reference_norm)
        if reference_norm else 0.0
    )
    atol = 5e-4 if dtype == torch.float32 else 2e-3
    return {
        "shape_equal": True,
        "exact": bool(torch.equal(first, second)),
        "numeric_close": bool(torch.allclose(first, second, atol=atol, rtol=1e-5)),
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "atol": atol,
    }


def visual_shape_key(inputs: dict) -> str:
    return digest_json({
        "pixel_shape": list(inputs["pixel_values"].shape),
        "grid": inputs["image_grid_thw"].detach().cpu().tolist(),
    })


def time_image(runtime, inputs: dict, modes: tuple[str, ...], image_number: int,
               warmed_modes: set[str], *, repeat_gate: bool) -> dict:
    shape_key = visual_shape_key(inputs)
    for mode in modes:
        if mode not in warmed_modes:
            feature, _, _ = method_runtime.timed_visual(runtime, inputs, mode)
            del feature
            warmed_modes.add(mode)
    forward = list(modes) if image_number % 2 == 0 else list(reversed(modes))
    sequence = [(mode, "A") for mode in forward]
    if repeat_gate:
        sequence += [(mode, "B") for mode in reversed(forward)]
    timings, features, profiles, repeat_checks = {}, {}, {}, {}
    for mode, half in sequence:
        feature, elapsed, profile = method_runtime.timed_visual(runtime, inputs, mode)
        timings[f"{mode}_{half}"] = elapsed
        serialized = jsonable(profile)
        if half == "A":
            features[mode] = feature
            profiles[mode] = serialized
        else:
            comparison = feature_comparison(
                features[mode], feature, runtime.model.visual.get_dtype()
            )
            comparison["profile_exact"] = (
                digest_json(serialized) == digest_json(profiles[mode])
            )
            repeat_checks[mode] = comparison
            if not comparison["numeric_close"] or not comparison["profile_exact"]:
                raise RuntimeError(f"same-path visual/route drift: {mode}/{comparison}")
            del feature
    for feature in features.values():
        del feature
    result = {
        "order": [f"{mode}_{half}" for mode, half in sequence],
        "timings_cuda_ms": timings,
        "mean_cuda_ms": {
            mode: (
                statistics.fmean((timings[f"{mode}_A"], timings[f"{mode}_B"]))
                if repeat_gate else timings[f"{mode}_A"]
            )
            for mode in modes
        },
        "profiles_A": profiles,
        "visual_shape_key": shape_key,
        "attention_backend": "eager",
        "timing_repetitions": 2 if repeat_gate else 1,
        "timing_protocol": (
            "bounded_smoke_A_B_stability"
            if repeat_gate else "smolvlm_formal_one_post_warmup_observation"
        ),
    }
    if repeat_gate:
        result["A_B"] = repeat_checks
    return result


def retry_three(operation):
    last_error = None
    for attempt in range(1, 4):
        try:
            return operation(), attempt
        except Exception:
            last_error = traceback.format_exc()
            if attempt == 3:
                error = RuntimeError("operation failed on all three attempts")
                error.last_traceback = last_error
                raise error
    raise AssertionError("unreachable")


def generate(runtime, inputs: dict, mode: str, generation: dict) -> dict:
    import torch

    if generation.get("natural_eos") is not True:
        raise RuntimeError("H300 quality requires the frozen natural-EOS contract")
    generation_kwargs = {
        key: value for key, value in generation.items() if key != "natural_eos"
    }
    with method_runtime.online_context(runtime, inputs, mode), torch.inference_mode():
        output = runtime.model.generate(
            **inputs,
            **generation_kwargs,
        )
        torch.cuda.synchronize()
    prefix = int(inputs["input_ids"].shape[1])
    new = output[:, prefix:]
    token_ids = [int(value) for value in new[0].detach().cpu().tolist()]
    text = runtime.processor.batch_decode(
        new, skip_special_tokens=True
    )[0].strip()
    return {
        "raw_text": text,
        "token_ids": token_ids,
        "generated_tokens": len(token_ids),
        "hit_cap": len(token_ids) >= int(generation_kwargs["max_new_tokens"]),
    }


def validate_generation_result(generated: dict, phase: str) -> None:
    """Reject an invalid migration smoke before it can look complete."""
    if phase != "smoke":
        return
    if (not generated["token_ids"] or not generated["raw_text"]
            or generated["hit_cap"]):
        first_token = generated["token_ids"][0] if generated["token_ids"] else None
        raise RuntimeError(
            "smoke generation invalid: "
            f"hit_cap={generated['hit_cap']}, "
            f"generated_tokens={generated['generated_tokens']}, "
            f"first_token={first_token}"
        )


def grouped_rows(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    groups = collections.OrderedDict()
    for row in rows:
        groups.setdefault(row["image_key"], []).append(row)
    return list(groups.items())


def selected_datasets(phase: str) -> tuple[str, ...]:
    if phase == "formal":
        return grid.DATASETS
    return ("HRBench4K", "MMMU_val")


def selected_groups(manifest: dict, phase: str):
    groups = grouped_rows(manifest["rows"])
    if phase == "formal":
        return groups
    chosen = []
    if manifest["dataset"] == "MMMU_val":
        multi = next(
            ((key, rows) for key, rows in groups if len(rows[0]["images"]) > 1),
            None,
        )
        if multi is None:
            raise RuntimeError("MMMU smoke manifest contains no multi-image request")
        chosen.append((multi[0], multi[1][:1]))
    for key, rows in groups:
        if key in {value[0] for value in chosen}:
            continue
        chosen.append((key, rows[:1]))
        if len(chosen) == 2:
            break
    return chosen


def journal_contract(overall: dict, manifest: dict, kind: str,
                     mode: str | None = None) -> dict:
    value = {
        "experiment": EXPERIMENT,
        "release": RELEASE,
        "phase": overall["phase"],
        "job": overall["job"],
        "method": overall["method"],
        "dtype": overall["dtype"],
        "dataset": manifest["dataset"],
        "manifest_digest": manifest["content_digest"],
        "kind": kind,
        "attention_backend": "eager",
        "source_files": overall["source_files"],
    }
    if mode is not None:
        value["mode"] = mode
    return value


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    with temporary.open("xb") as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def materialize_dataset(output: Path, manifest: dict,
                        groups: list[tuple[str, list[dict]]],
                        modes: tuple[str, ...], performance: Journal,
                        quality: dict[str, Journal]) -> tuple[list[dict], list[dict]]:
    performance_rows = [
        performance.result(image_key)
        for image_key, _ in groups if performance.contains(image_key)
    ]
    quality_rows = []
    for _, rows in groups:
        for row in rows:
            request_id = str(row["id"])
            if not all(quality[mode].contains(request_id) for mode in modes):
                continue
            quality_rows.append({
                "dataset": manifest["dataset"],
                "id": request_id,
                "image_key": row["image_key"],
                "prompt_sha256": row["prompt_sha256"],
                "execution": "online_visual_per_question",
                "modes": {
                    mode: quality[mode].result(request_id)["output"]
                    | {"attempts": quality[mode].result(request_id)["attempts"]}
                    for mode in modes
                },
            })
    atomic_jsonl(output / f"{manifest['dataset']}.performance.jsonl", performance_rows)
    atomic_jsonl(output / f"{manifest['dataset']}.quality.jsonl", quality_rows)
    return performance_rows, quality_rows


def generation_cap_statistics(rows: list[dict], modes: tuple[str, ...]) -> dict:
    by_mode = {
        mode: sum(bool(row.get("modes", {}).get(mode, {}).get("hit_cap")) for row in rows)
        for mode in modes
    }
    return {
        "total": sum(by_mode.values()),
        "by_mode": by_mode,
        "affected_rows": sum(
            any(value.get("hit_cap") for value in row.get("modes", {}).values())
            for row in rows
        ),
    }


def existing_committed_count(output: Path) -> int:
    count = 0
    journal_root = output / "journals"
    paths = journal_root.rglob("segment_*.jsonl") if journal_root.is_dir() else ()
    for path in paths:
        with path.open("rb") as stream:
            count += sum(1 for line in stream if line.endswith(b"\n"))
    return count


def expected_units(manifests: dict[str, dict], phase: str,
                   modes: tuple[str, ...]) -> int:
    total = 0
    for manifest in manifests.values():
        groups = selected_groups(manifest, phase)
        total += len(groups) + sum(len(rows) for _, rows in groups) * len(modes)
    return total


def status_snapshot(output: Path, state: str, current: dict | None,
                    committed: int, expected: int, started: float,
                    deadline: float) -> None:
    atomic_json(output / "STATUS.json", {
        "experiment": EXPERIMENT,
        "release": RELEASE,
        "state": state,
        "expected_units": expected,
        "committed_units": committed,
        "missing_units": expected - committed,
        "current": current,
        "elapsed_seconds_this_epoch": time.monotonic() - started,
        "remaining_seconds_this_epoch": max(0, deadline - time.monotonic()),
        "updated_unix": time.time(),
    })


def run(args) -> dict:
    import torch

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    method, dtype_name = grid.JOB_SPECS[args.job]
    modes = grid.modes(method)
    datasets = selected_datasets(args.phase)
    manifests = {
        dataset: load_manifest(args.manifest_dir / f"{dataset}.json", args.phase)
        for dataset in datasets
    }
    timing_order = {
        (dataset, image_key): index
        for index, (dataset, image_key) in enumerate(
            (dataset, image_key)
            for dataset in datasets
            for image_key, _rows in selected_groups(manifests[dataset], args.phase)
        )
    }
    snapshot_receipt = verify_snapshot(args.snapshot)
    output = args.output_root / f"job_{args.job}"
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "experiment": EXPERIMENT,
        "release": RELEASE,
        "phase": args.phase,
        "job": args.job,
        "method": method,
        "dtype": dtype_name,
        "attention_backend": "eager",
        "quality_execution": "online_visual_per_question",
        "modes": list(modes),
        "datasets": list(datasets),
        "protocol": grid.PROTOCOL,
        "configs": grid.CONFIGS[method],
        "manifests": {key: value["content_digest"] for key, value in manifests.items()},
        "snapshot": snapshot_receipt,
        "source_files": source_files(),
    }
    ensure_contract(output / "CONTRACT.json", contract)

    runtime = method_runtime.load(
        method, dtype_name, args.snapshot, output / "scratch", args.phase
    )
    atomic_json(output / "RUNTIME.json", runtime.receipt)
    torch.set_grad_enabled(False)
    started = time.monotonic()
    deadline = started + args.max_seconds
    expected = expected_units(manifests, args.phase, modes)
    committed = existing_committed_count(output)
    current = None
    status_snapshot(output, "starting", current, committed, expected, started, deadline)
    timing_warmed_modes: set[str] = set()
    warmed_modes: set[str] = set()
    dataset_status = {}
    paused = False
    epoch = uuid.uuid4().hex
    try:
        for dataset in datasets:
            manifest = manifests[dataset]
            groups = selected_groups(manifest, args.phase)
            # Preserve SmolVLM's frozen global request order when rotating mode
            # execution.  Using the row's position within one image cluster would
            # repeatedly restart at zero and bias the first mode on datasets that
            # contain many questions per image.
            manifest_index = {
                str(row["id"]): index
                for index, row in enumerate(manifest["rows"])
            }
            expected_ids = [str(row["id"]) for _, rows in groups for row in rows]
            expected_images = [image_key for image_key, _ in groups]
            with contextlib.ExitStack() as stack:
                performance = stack.enter_context(Journal(
                    output / "journals" / "performance" / dataset,
                    journal_contract(contract, manifest, "performance"),
                    segment_records=256,
                ))
                quality = {
                    mode: stack.enter_context(Journal(
                        output / "journals" / "quality" / mode / dataset,
                        journal_contract(contract, manifest, "quality", mode),
                        segment_records=256,
                    ))
                    for mode in modes
                }
                performance.missing(expected_images)
                for journal in quality.values():
                    journal.missing(expected_ids)

                for image_key, rows in groups:
                    pending_timing = not performance.contains(image_key)
                    pending_quality = any(
                        not quality[mode].contains(str(row["id"]))
                        for row in rows for mode in modes
                    )
                    if not pending_timing and not pending_quality:
                        continue
                    if time.monotonic() >= deadline:
                        paused = True
                        break
                    if pending_timing:
                        representative = rows[0]
                        inputs = prepare_inputs(runtime, representative, args.data_root)
                        timing_row = time_image(
                            runtime, inputs, modes, timing_order[(dataset, image_key)],
                            timing_warmed_modes,
                            repeat_gate=args.phase != "formal",
                        )
                        timing_row.update({
                            "dataset": dataset,
                            "image_key": image_key,
                            "representative_id": str(representative["id"]),
                            "image_count": len(representative["images"]),
                            "visual_input_tokens": int(inputs["pixel_values"].shape[0]),
                            "epoch": epoch,
                        })
                        if performance.commit(image_key, timing_row):
                            committed += 1
                        del inputs

                    for row in rows:
                        request_id = str(row["id"])
                        pending_modes = [
                            mode for mode in modes if not quality[mode].contains(request_id)
                        ]
                        if not pending_modes:
                            continue
                        inputs = prepare_inputs(runtime, row, args.data_root)
                        rotation = manifest_index[request_id] % len(modes)
                        ordered_modes = modes[rotation:] + modes[:rotation]
                        for mode in ordered_modes:
                            if quality[mode].contains(request_id):
                                continue
                            if time.monotonic() >= deadline:
                                paused = True
                                break
                            current = {
                                "dataset": dataset,
                                "image_key": image_key,
                                "request_id": request_id,
                                "mode": mode,
                            }
                            if mode not in warmed_modes:
                                with method_runtime.online_context(runtime, inputs, mode), \
                                        torch.inference_mode():
                                    runtime.model.generate(
                                        **inputs,
                                        do_sample=False,
                                        num_beams=1,
                                        max_new_tokens=1,
                                        use_cache=True,
                                    )
                                    torch.cuda.synchronize()
                                warmed_modes.add(mode)
                            try:
                                generated, attempts = retry_three(
                                    lambda mode=mode: generate(
                                        runtime, inputs, mode, manifest["generation"]
                                    )
                                )
                            except Exception as exc:
                                atomic_json(
                                    output / "attempt_failures"
                                    / f"{digest_json(current | {'epoch': epoch})}.json",
                                    current | {
                                        "epoch": epoch,
                                        "attempts": 3,
                                        "error": getattr(exc, "last_traceback", traceback.format_exc()),
                                    },
                                )
                                raise RuntimeError(
                                    f"quality unit failed without commit: {current}"
                                ) from exc
                            validate_generation_result(generated, args.phase)
                            result = {
                                "dataset": dataset,
                                "request_id": request_id,
                                "image_key": image_key,
                                "prompt_sha256": row["prompt_sha256"],
                                "mode": mode,
                                "epoch": epoch,
                                "attention_backend": "eager",
                                "execution": "online_visual_per_question",
                                "attempts": attempts,
                                "output": generated,
                            }
                            if quality[mode].commit(request_id, result):
                                committed += 1
                            if generated["hit_cap"]:
                                print(json.dumps({
                                    "event": "generation_safety_cap_recorded",
                                    "dataset": dataset,
                                    "id": request_id,
                                    "mode": mode,
                                    "cap": grid.PROTOCOL["generation_safety_cap"],
                                    "action": "continue_and_score_recorded_output",
                                }, ensure_ascii=False), flush=True)
                            if committed % 25 == 0:
                                status_snapshot(
                                    output, "running", current, committed, expected,
                                    started, deadline,
                                )
                                print(json.dumps({
                                    "committed": committed,
                                    "expected": expected,
                                    **current,
                                }, ensure_ascii=False), flush=True)
                        del inputs
                        if paused:
                            break
                    if paused:
                        break

                performance_rows, quality_rows = materialize_dataset(
                    output, manifest, groups, modes, performance, quality
                )
                missing_timing = performance.missing(expected_images)
                missing_by_mode = {
                    mode: quality[mode].missing(expected_ids) for mode in modes
                }
                dataset_status[dataset] = {
                    "questions": len(quality_rows),
                    "expected_questions": len(expected_ids),
                    "images": len(performance_rows),
                    "expected_images": len(expected_images),
                    "quality_execution": "online_visual_per_question",
                    "attention_backend": "eager",
                    "generation_cap_hits": generation_cap_statistics(quality_rows, modes),
                    "missing_timing": len(missing_timing),
                    "missing_quality_by_mode": {
                        mode: len(missing) for mode, missing in missing_by_mode.items()
                    },
                    "complete": not missing_timing and all(
                        not missing for missing in missing_by_mode.values()
                    ),
                }
                atomic_json(
                    output / "dataset_status" / f"{dataset}.json",
                    dataset_status[dataset],
                )
            status_snapshot(
                output, "paused_budget" if paused else "running", current,
                committed, expected, started, deadline,
            )
            if paused:
                break

        state = "paused_budget" if paused else "complete"
        if not paused and (
            set(dataset_status) != set(datasets)
            or not all(value["complete"] for value in dataset_status.values())
            or committed != expected
        ):
            raise RuntimeError("job coverage incomplete")
        final = {
            "state": state,
            "experiment": EXPERIMENT,
            "release": RELEASE,
            "phase": args.phase,
            "job": args.job,
            "method": method,
            "dtype": dtype_name,
            "attention_backend": "eager",
            "quality_execution": "online_visual_per_question",
            "expected_units": expected,
            "committed_units": committed,
            "datasets": dataset_status,
            "wall_seconds": time.monotonic() - started,
            "max_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        atomic_json(output / "FINAL_STATUS.json", final)
        status_snapshot(output, state, current, committed, expected, started, deadline)
        print(json.dumps(final, ensure_ascii=False), flush=True)
        return final
    except BaseException as exc:
        status_snapshot(output, "failed", current, committed, expected, started, deadline)
        atomic_json(output / "FINAL_STATUS.json", {
            "state": "failed",
            "experiment": EXPERIMENT,
            "release": RELEASE,
            "phase": args.phase,
            "job": args.job,
            "method": method,
            "dtype": dtype_name,
            "attention_backend": "eager",
            "quality_execution": "online_visual_per_question",
            "expected_units": expected,
            "committed_units": committed,
            "datasets": dataset_status,
            "error": repr(exc),
            "wall_seconds": time.monotonic() - started,
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--job", type=int, choices=range(10), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=259200)
    args = parser.parse_args()
    result = run(args)
    raise SystemExit(0 if result["state"] == "complete" else 75)


if __name__ == "__main__":
    main()
