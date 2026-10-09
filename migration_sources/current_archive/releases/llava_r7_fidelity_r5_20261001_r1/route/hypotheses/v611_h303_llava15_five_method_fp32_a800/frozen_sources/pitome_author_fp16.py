# Publication copy: comments/line endings only; algorithm AST unchanged.
# Project migration copy; retain upstream method and host notices. No experiment executed.
# Component licensing and exact source hashes: sources/FILE_DISPOSITIONS.json.
"""Kaggle T4x2 gate for the frozen author PiToMe LLaVA/CLIP path."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
import zlib
from pathlib import Path

OUT = Path(os.environ["H297_OUTPUT"])
OUT.mkdir(parents=True, exist_ok=True)

AUTHOR_SHA256 = {'merge.py': '57f286663de05b1ce6a239f0db015cf745475a8c271e4ac0d214dc3e521b6145', 'patch/clip_hf.py': 'a32b5db7e78b9e1c49cf7672d34ef75993ebbd4d8ff965828e97c90a071ea57a'}
AUTHOR_PAYLOADS = {'merge.py': 'eNrlWm1v4zYS/r7A/oe5BAdIOVmxg/aLcTp0G3QPBdri0OZwH4LAYCTa5kWitCQV2/31N3yTKVl2vNtst4cGC69MzgzJeXlmNPQl3NbNTrDVWkGUx/AjVQT+VRK1rEUlE/ie5ykQXgBZLlnJiKIyffvmEt6VJRguCYJKKp5pgeN65m7NJMi6FTmFvC4o4NeS5ZRLWkDLCypArakfgmWNY8C4HtTsP3x/+91Pv3wHuBp1wyDqWkHBBM1VLXZQL3F0v4gSlJo9TT7xT++bVU0tFFRErd++WYq6ArVrGF+Bm7glZUkeS5rAXduUtGPADeXr/reUcyASOD8YTpctzxWrOSk1xfuOgLdVszNMjd7M2zcFXUJRL3iNB+WraJtAharMfqo5jedv3wD+CapawWHryR9lFdmZiirB8rlb9g7VXIvEMRFcfr4sa6KyWTpN9JidyUsi5ULVT5TP4bGuS8jgPSklRc4YJv+w577f68E/Pbj92M9G1ArNhKbOYGqH2LIn3A72af+WwawjLimP7BFSuSYNjSHL4Cbgs5O4gH2411pJ0jR90KrQBJfwH/Q9wqHm5Q4VVbToJo+oYLTvllVtpV3o6+lfwexIWqa7TqBd9n72EB4MN2aUB38HVFywG6H50G1S1GotorsJ3F0ZytjSUNRhSG7NtjduEjxb42uyDVNr7zf1YiVIEcXHVQDXfuscwzYqWJVNZgk8Udro5zvR0njPTBJ43GsPFZfAfH6DHw9Jb3DmRvecMq8x2pGXwDfwmCpBuGxqSSO92uQm9rs/afa9ILvMVK+BMidGi4wvQykc3X6BVkvsEyu2SGm5Uxx2Rw0OR4uVJ/O8KRErDAHV6aWgEsGnQIVb1dh9aDd6CNdueeUkeaGWUMzNjtHN/s0rKlbowHeBI5nziXyUdS4864/jjIVU4e47RrO5dIUaosKeA03DEE23mVvrfOXvz+We0k47s/h+2ulAo4o5X7QdgIlFo4uKEn5hwCGcnfe2oaN5eySQ9d82bbn80FL6K11E07inwkSrA7e5HTjp9ph/8gQU2lcHBrLbVXv2dBOjevTKoNsGE15kZMEE8LR53N/XKSnOGoGUoQB7JvxMZU6UomJhESrSQpz9h+yJVYclzLT2e+Z2oGKtgEKjezyMEYbasnb1WObZHIuxr08hDVN1RXtZxKSc5OUUYayNIZVTOQ+dwfI7DLVxOzLvtjRnXJmBM1POt5iNEwPcDhECg1+CpA1FIKZBnjF7N+WMObeW5XKAh3kz7qLDncc5382VeBjkuDE6JPMeSfQ8Yq0j60R33mzwtjeoXfoB9nlsRRVIVrGSCKZ2Hn8fqdpQyg8OEWp5r5TQTWfeTc2m9qGHvhd7n7NaFXEML8vr3J4M5c1CeejDAnwMLJIA5Q6hHMKs+yoQdBwJt4u8lNn2fp5Mk/lDsP0woxhCQzSbJ/NehrCqutUQNQSbR6LytTuk3RrBbIkH+TYOEQ/VpGpf7hT0GR0pHquQNAh2IpO+Cw5ydIibAQvxpEl/3PpnXwYqTasX/pLBRSNaTi8GijsLw/bqvBk4w+1LgHbSaodoZ+wYaGUM+sY5j7KMo2SIk4thxf06WPn6UHlWRWldeTGKpSNg1sdGA2W9IQ9krxoOr4NslsDW+JPZQ4dzr4R1pC9+ZL0vWd3+n9e2f/B88LHYOwK9v0cJ+yK8/+Ya9zPVuZ+WFo7Uwh9VOB8vlxfPTLKan+q9wHjzxXFg/DLeR/pp+vWZeYSUzZoE0lpp8lIW0rixhd1vN+XRH5XqKA7bFDrFueRm/0vsKTLzmYTby4Ln+KyUc9KavVX3sbZPV3fJ7aBj05M83qw5p2Ez0rQ5q3HjCS8hJ2XelvrVg3J0GFe3A4z0b96bjg0pGYKM13CT3Vjn21fDJihZ1R34G3fuXgfGNGC6E5iVF3ZlvQ4t2yjSMibO5TDYrPeYzzjV0HmYcVxWRxFd3rbpJlzgMOn0bNF3wHG9BjXVwOGO+FhXcGTu/8RlzgwPiW6aibPs6d55X3HNgzpR0a36AvDgQ+0zROFBh+dkv/aFnu15IX1O1H5KVL1KYXqJtc6RqJewrAVoF3Av6Ym91WBSXy2UtKJc6WOgb22ojhRz4/FP0krJCIcnKjgtU3ersq435u0fkyk0VOg7GsL1Fcia2JsS+qE1wiD6Kva3Jw3qUXwyjlw6egcgp9n22HI9TaezPsCEO1hVBGXOOno4ClyRNQrWCdEkiiLNghuKr42M+OrqBq4A4yGOHYBp614BzK4jQ3Fl2eUHRCz7qEwMRTdupmFx7w3gLLyDU4B3TgvpY9pI1grvSlkjiVSUFPr6opX6dqouCtPOos+UQ4dJ6Ej5uq61L9Ugm5IpdHN9/YaOsWYrLOgMU1lv8Mk7K1USNmuWr42baRtxtmQ54arcjfnc0VfDYZ9LHLa5xOilAtr1NzerPkfD6rym1R/4BeVEFjQC8Ui2lXBWZ+tLNaX+3D0nG0EG4oJO9Ty4mB44XK8x1PfE8JvvpV9cXNiHdw0CBoaPTh1mDfC31voKVZEnjTsEyDMVBCc7YLG9dYkpN7WCfjbnCAQVYLHXgI8e5XQTsMl0sJMOTLr7bxc94xfg5k32z6wc+yo/op3QgzbkeXWmkhKzcn+sg4rPqr8N1T8tQZ38DorEsNfj+mcq+mTzsEz5lXZVIU7JRcmeUN/dtbVpXHVxG1jkSrN6s8i28kZxEi3VERotZgvXhnYACVtrERja1P4Q5nyrGvJTdh1JTZ2+3iNx98sbkmuLpfCLHUAdYiFKiv8SrI3ynS7YBdt291XaGowzxUjpr920kZZM/yDGGW8l6rbxRtpqia00N3eaDF2gVaDLE6xBdl6Gzu4UiDC/F8qJK6FltyV9sDGr9wjm/RacbYUf5D7H452C7mikdC2o3weyrX8xsG83oF9vwlYc/uucpRPkfMF89d5AsLIYxLEl6JtdJ0GuY2dREfkUmh8Ta2/uAPzmrlzqk3Xb6Q+f2NaA3zrm/wDUd1TF', 'patch/clip_hf.py': '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'}
AUTHOR_ROOT = OUT / 'author_pitome'
def materialize_author_sources():
    AUTHOR_ROOT.mkdir(parents=True, exist_ok=True)
    (AUTHOR_ROOT / '__init__.py').write_text('', encoding='utf-8')
    (AUTHOR_ROOT / 'patch').mkdir(parents=True, exist_ok=True)
    (AUTHOR_ROOT / 'patch/__init__.py').write_text('', encoding='utf-8')
    for name, payload in AUTHOR_PAYLOADS.items():
        source = zlib.decompress(base64.b64decode(payload))
        if hashlib.sha256(source).hexdigest() != AUTHOR_SHA256[name]: raise RuntimeError(name)
        target = AUTHOR_ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        stream.flush()


def load_images(limit=8):
    from PIL import Image
    paths = [Path(os.environ["H297_POPE"])]
    if len(paths) != 1:
        raise RuntimeError(f"expected one POPE.tsv, got {paths}")
    csv.field_size_limit(2**31 - 1)
    rows, seen = [], set()
    with paths[0].open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            digest = hashlib.sha256(row["image"].encode("ascii")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            image = Image.open(io.BytesIO(base64.b64decode(row["image"]))).convert("RGB")
            rows.append((str(row["index"]), digest, image))
            if len(rows) == limit:
                break
    if len(rows) != limit:
        raise RuntimeError(f"expected {limit} images, got {len(rows)}")
    return rows


def worker(rank):
    import torch
    import transformers
    from transformers import CLIPImageProcessor, CLIPVisionModel
    from llava_vision_loader import load as load_llava_vision

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dtype = torch.float16 if rank == 0 else torch.float32
    dtype_name = "float16" if rank == 0 else "float32"
    target = OUT / f"worker_{rank}"
    materialize_author_sources()
    sys.path.insert(0, str(AUTHOR_ROOT.parent))
    import author_pitome.patch.clip_hf as author_clip_patch

    snapshot = Path(os.environ["H297_LLAVA_SNAPSHOT"])
    processor = CLIPImageProcessor.from_pretrained(snapshot, local_files_only=True)
    dense = load_llava_vision(snapshot, torch=torch, dtype=dtype)
    candidate = load_llava_vision(snapshot, torch=torch, dtype=dtype)
    author_clip_patch.apply_patch(candidate.vision_model.encoder)

    # Observe the author's discrete merge closure without changing its implementation.
    # The one-hot probe records exact source/destination membership and ordering.
    original_pitome_vision = author_clip_patch.pitome_vision
    route_state = {"enabled": False, "layers": []}

    def observed_pitome_vision(*args, **kwargs):
        merge = original_pitome_vision(*args, **kwargs)
        if route_state["enabled"]:
            metric = kwargs.get("metric", args[0] if args else None)
            tokens = int(metric.shape[-2])
            identity = torch.eye(tokens, device=metric.device, dtype=metric.dtype).unsqueeze(0)
            membership = merge(identity, mode="sum")
            route_state["layers"].append({
                "input_tokens": tokens,
                "output_tokens": int(membership.shape[-2]),
                "membership_sha256": hashlib.sha256(
                    membership.detach().float().cpu().numpy().tobytes()
                ).hexdigest(),
            })
        return merge

    author_clip_patch.pitome_vision = observed_pitome_vision

    runtime = {
        "rank": rank,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dtype": dtype_name,
        "tf32": False,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "fp16_reduced_precision_reduction": False,
        "author_repository": "https://github.com/hchautran/PiToMe",
        "author_commit": "550b5deed94aadfeac28bfbe381d87e672044a40",
        "author_source_sha256": AUTHOR_SHA256,
        "model_id": "openai/clip-vit-large-patch14-336",
        "model_revision": "17011ea7a0975c9c99b1c985d7b3b8a78e907c1e",
        "boundary": "author PiToMe patched complete CLIP tower; LLaVA penultimate hidden state; no projector/LM",
    }
    save(target / "RUNTIME.json", runtime)
    save(target / "CONTRACT.json", {
        "state": "frozen",
        "ratios": [1.0, 0.98, 0.96],
        "images": int(os.environ.get("H297_IMAGES", "32")),
        "native_path": "algo/pitome/patch/clip_hf.py applied exactly as tasks/vqa/lmms_eval/models/llava.py",
        "required": ["disabled_exact", "active_shorter", "same_decisions_exact", "numeric_stability", "balanced_ab"],
        "runtime": runtime,
    })

    def pixels(image):
        return processor(images=image, return_tensors="pt")["pixel_values"].to("cuda", dtype)

    def run(model, pv, ratio=None):
        if ratio is not None:
            model.vision_model.encoder.ratio = ratio
        result = model(pixel_values=pv, output_hidden_states=True, return_dict=True)
        value = result.hidden_states[-2][:, 1:]
        trace = [int(state.shape[1]) for state in result.hidden_states]
        return value, trace

    def route_run(model, pv, ratio):
        route_state["enabled"] = True
        route_state["layers"] = []
        try:
            value, trace = run(model, pv, ratio)
            torch.cuda.synchronize()
            return value.detach().clone(), trace, list(route_state["layers"])
        finally:
            route_state["enabled"] = False

    def difference(left, right):
        delta = (left.float() - right.float()).reshape(-1)
        max_abs = float(delta.abs().max().item())
        relative_l2 = float(delta.norm().item() / max(left.float().reshape(-1).norm().item(), 1e-12))
        return max_abs, relative_l2

    def timed(call):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        begin.record()
        value, trace = call()
        end.record()
        end.synchronize()
        return value.detach().clone(), trace, float(begin.elapsed_time(end))

    images = load_images(int(os.environ.get("H297_IMAGES", "32")))
    modes = ("dense", "disabled", "r098", "r096")
    ratios = {"disabled": 1.0, "r098": 0.98, "r096": 0.96}
    with torch.inference_mode():
        for image_number, (sample_id, image_sha, image) in enumerate(images):
            pv = pixels(image)
            run(dense, pv)
            for warm_ratio in (1.0, 0.98, 0.96):
                run(candidate, pv, warm_ratio)
            torch.cuda.synchronize()
            for mode in ("r098", "r096"):
                route_a_value, route_a_trace, route_a = route_run(candidate, pv, ratios[mode])
                route_b_value, route_b_trace, route_b = route_run(candidate, pv, ratios[mode])
                max_abs, relative_l2 = difference(route_a_value, route_b_value)
                route_exact = route_a == route_b
                trace_exact = route_a_trace == route_b_trace
                numeric_limit = 1e-3 if dtype == torch.float16 else 1e-4
                append(target / "ROUTE_AUDIT.jsonl", {
                    "rank": rank,
                    "image_number": image_number,
                    "sample_id": sample_id,
                    "image_sha256": image_sha,
                    "mode": mode,
                    "route_exact": route_exact,
                    "trace_exact": trace_exact,
                    "route": route_a,
                    "max_abs": max_abs,
                    "relative_l2": relative_l2,
                    "relative_l2_limit": numeric_limit,
                })
                if not route_exact or not trace_exact or relative_l2 > numeric_limit:
                    raise RuntimeError(
                        f"route/numeric drift: {sample_id}/{mode}; route={route_exact}; "
                        f"trace={trace_exact}; max_abs={max_abs}; relative_l2={relative_l2}"
                    )
            first = {}
            order = modes if image_number % 2 == 0 else tuple(reversed(modes))
            for half, sequence in (("A", order), ("B", tuple(reversed(order)))):
                for mode in sequence:
                    if mode == "dense":
                        value, trace, ms = timed(lambda: run(dense, pv))
                    else:
                        value, trace, ms = timed(lambda m=mode: run(candidate, pv, ratios[m]))
                    if mode in first:
                        old_value, old_trace = first[mode]
                        max_abs, relative_l2 = difference(old_value, value)
                        trace_exact = old_trace == trace
                        numeric_limit = 1e-3 if dtype == torch.float16 else 1e-4
                        append(target / "STABILITY.jsonl", {
                            "rank": rank,
                            "image_number": image_number,
                            "sample_id": sample_id,
                            "image_sha256": image_sha,
                            "mode": mode,
                            "trace_exact": trace_exact,
                            "max_abs": max_abs,
                            "relative_l2": relative_l2,
                            "relative_l2_limit": numeric_limit,
                        })
                        if not trace_exact or relative_l2 > numeric_limit:
                            raise RuntimeError(
                                f"timed-path drift: {sample_id}/{mode}; trace={trace_exact}; "
                                f"max_abs={max_abs}; relative_l2={relative_l2}"
                            )
                    first.setdefault(mode, (value, trace))
                    append(target / "RAW.jsonl", {
                        "rank": rank,
                        "image_number": image_number,
                        "sample_id": sample_id,
                        "image_sha256": image_sha,
                        "mode": mode,
                        "half": half,
                        "cuda_ms": ms,
                        "selected_tokens": int(value.shape[1]),
                        "hidden_state_tokens": trace,
                    })
            dense_value, dense_trace = first["dense"]
            disabled_value, disabled_trace = first["disabled"]
            disabled_exact = bool(torch.equal(dense_value, disabled_value) and dense_trace == disabled_trace)
            active_shorter = {
                mode: bool(first[mode][0].shape[1] < dense_value.shape[1]) for mode in ("r098", "r096")
            }
            append(target / "FEATURE_GATE.jsonl", {
                "sample_id": sample_id,
                "disabled_exact": disabled_exact,
                "disabled_max_abs": float((dense_value - disabled_value).abs().max().item()),
                "active_shorter": active_shorter,
                "dense_tokens": int(dense_value.shape[1]),
                "active_tokens": {mode: int(first[mode][0].shape[1]) for mode in active_shorter},
            })
            if not disabled_exact or not all(active_shorter.values()):
                raise RuntimeError(f"author-path gate failed: {sample_id}")
            print(json.dumps({"rank": rank, "completed_images": image_number + 1, "total": int(os.environ.get("H297_IMAGES", "32"))}), flush=True)

    rows = [json.loads(line) for line in (target / "RAW.jsonl").read_text(encoding="utf-8").splitlines()]
    means = {mode: statistics.fmean(row["cuda_ms"] for row in rows if row["mode"] == mode) for mode in modes}
    summary = {
        "state": "complete",
        "rank": rank,
        "dtype": dtype_name,
        "images": int(os.environ.get("H297_IMAGES", "32")),
        "records": len(rows),
        "disabled_exact": True,
        "active_shorter": True,
        "same_decisions_exact": True,
        "numeric_stability": True,
        "mean_cuda_ms": means,
        "dense_over_candidate": {mode: means["dense"] / means[mode] for mode in ("r098", "r096")},
        "timing_role": "formal V100 FP16 complete-vision-tower timing",
    }
    save(target / "SUMMARY.json", summary)
    save(target / "FINAL_STATUS.json", summary)


def parent():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.37.0"], check=True)
    import torch
    import transformers
    from huggingface_hub import snapshot_download

    if torch.cuda.device_count() != 2 or any("T4" not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError(f"requires T4x2, got {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    if transformers.__version__ != "4.37.0":
        raise RuntimeError(transformers.__version__)
    snapshot = snapshot_download(
        "openai/clip-vit-large-patch14-336",
        revision="17011ea7a0975c9c99b1c985d7b3b8a78e907c1e",
        allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
    )
    processes, logs = [], []
    for rank in (0, 1):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        env["H287_CLIP_SNAPSHOT"] = snapshot
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        log = (OUT / f"worker_{rank}.log").open("w", encoding="utf-8")
        logs.append(log)
        processes.append(subprocess.Popen([sys.executable, __file__, "--worker", str(rank)], env=env,
                                          stdout=log, stderr=subprocess.STDOUT))
    codes = [process.wait() for process in processes]
    for log in logs:
        log.close()
    status = {
        "state": "complete" if codes == [0, 0] else "failed",
        "exit_codes": codes,
        "workers": 2,
        "author_commit": "550b5deed94aadfeac28bfbe381d87e672044a40",
    }
    save(OUT / "FINAL_STATUS.json", status)
    if any(codes):
        raise RuntimeError(f"workers failed: {codes}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--worker", type=int)
        args = parser.parse_args()
        worker(args.worker) if args.worker is not None else parent()
    except Exception as exc:
        save(OUT / "FINAL_STATUS.json", {"state": "failed", "error": repr(exc),
                                         "traceback": traceback.format_exc()})
        raise

