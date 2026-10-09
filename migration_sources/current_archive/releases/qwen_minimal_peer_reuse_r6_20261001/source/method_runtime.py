# Publication copy: comments/line endings only; algorithm AST unchanged.
# Project migration copy; retain upstream method and host notices. No experiment executed.
# Component licensing and exact source hashes: sources/FILE_DISPOSITIONS.json.
"""Unified eager Qwen runtime for H300's five admitted methods."""

from __future__ import annotations

import contextlib
import importlib.metadata
import sys
from dataclasses import dataclass
from pathlib import Path

import grid
import h259_bridge
import fp16_eager_stability
from qwen_peer_ports import PeerConfig, QwenPeerVisionPort


MODEL_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
NATIVE_METHODS = {"ours", "ipcv", "illava"}
HOST_PORTS = {"dtome", "pitome"}


def forbid_sdpa_operator() -> None:
    """Make the strict eager contract executable for every loaded path.

    H300 runs in a dedicated process.  Replacing the functional SDPA entry
    point here makes any hidden fallback in a native, peer, or host adapter
    fail during the Kaggle preflight instead of silently changing the formal
    V100 execution backend.
    """

    import torch.nn.functional as functional

    current = functional.scaled_dot_product_attention
    if getattr(current, "_h300_forbidden_sdpa", False):
        return

    def forbidden(*_args, **_kwargs):
        raise RuntimeError(
            "H300 strict SmolVLM alignment forbids scaled_dot_product_attention"
        )

    forbidden._h300_forbidden_sdpa = True
    functional.scaled_dot_product_attention = forbidden


@dataclass
class Runtime:
    method: str
    dtype_name: str
    source: object
    model: object
    processor: object
    receipt: dict


def environment_gate(phase: str) -> dict:
    import torch

    transformers_version = importlib.metadata.version("transformers")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    if phase == "formal" and not torch.__version__.startswith("2.5.1+cu121"):
        raise RuntimeError(f"expected torch 2.5.1+cu121, got {torch.__version__}")
    if transformers_version != "4.57.6":
        raise RuntimeError(f"expected common-stack Transformers 4.57.6, got {transformers_version}")
    if gpu is None or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible GPU, got {gpu}")
    if phase == "formal" and (
        "V100" not in gpu or torch.cuda.get_device_properties(0).total_memory < 30 * 1024**3
    ):
        raise RuntimeError(f"formal H300 requires one 32-GiB V100, got {gpu}")
    # Match the formal SmolVLM runtime policy rather than inheriting whatever
    # thread/determinism state the host happens to expose.
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    forbid_sdpa_operator()
    return {
        "torch": torch.__version__, "transformers": transformers_version,
        "cuda": torch.version.cuda, "gpu": gpu, "tf32": False,
        "phase": phase, "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "attention_backend": "eager", "batch_size": 1,
        "sdpa_operator_forbidden": True,
    }


def _loading_summary(loading: dict) -> dict:
    keys = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    result = {key: [str(value) for value in loading.get(key, [])] for key in keys}
    if any(result.values()):
        raise RuntimeError(f"checkpoint loading changed: {result}")
    return result


def ensure_ipcv_vision_initializer(config) -> dict:
    """Bind the 4.57 weight-init API field without changing IPCV math.

    IPCV's vendored Qwen2-VL vision config predates the common 4.57 loader and
    therefore does not declare ``initializer_range``.  Transformers 4.57 asks
    every nested config for the field while finalizing checkpoint loading,
    even when the checkpoint itself supplies the model weights.  The matching
    value already exists on IPCV's top-level Qwen config, so expose that value
    to the nested vision config rather than inventing a new parameter.
    """

    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        raise RuntimeError("IPCV config has no vision_config")
    if hasattr(vision_config, "initializer_range"):
        value = float(vision_config.initializer_range)
        source = "vision_config"
    else:
        if not hasattr(config, "initializer_range"):
            raise RuntimeError("IPCV config has no initializer_range")
        value = float(config.initializer_range)
        vision_config.initializer_range = value
        source = "top_level_config"
    return {"source": source, "value": value}


def ipcv_generation_compatible_class(model_class):
    """Give a pre-4.50 author model the public 4.57 generation API.

    Transformers 4.50 removed ``GenerationMixin`` from ``PreTrainedModel``.
    IPCV's frozen Qwen class still provides its own generation preparation and
    cache hooks, but it no longer receives the public ``generate`` method by
    inheritance.  Compose the unchanged author class with the official mixin,
    exactly as the Transformers migration warning prescribes.
    """

    from transformers.generation import GenerationMixin

    if issubclass(model_class, GenerationMixin):
        return model_class
    return type(
        f"{model_class.__name__}Common457GenerationCompat",
        (model_class, GenerationMixin),
        {
            "__module__": model_class.__module__,
            "__doc__": "Transformers 4.57 generation API compatibility only.",
        },
    )


def ensure_ipcv_legacy_cache_api() -> dict:
    """Expose the cache-length names used by IPCV's frozen 4.45 model.

    Transformers 4.57 kept the same cache state and update contract but renamed
    the legacy length helpers.  IPCV's author attention calls
    ``get_usable_length`` and its static-cache path calls ``get_max_length``.
    Recreate the former public semantics on the 4.57 ``Cache`` base class so
    every cache instance created by the official generation loop sees them.
    No cache tensor, eviction policy, or method decision is changed.
    """

    from transformers.cache_utils import Cache

    if not hasattr(Cache, "get_max_length"):
        def get_max_length(self):
            value = self.get_max_cache_shape()
            return None if value is None or int(value) < 0 else value

        Cache.get_max_length = get_max_length

    if not hasattr(Cache, "get_usable_length"):
        def get_usable_length(self, new_seq_length: int, layer_idx: int = 0):
            max_length = self.get_max_length()
            previous_seq_length = self.get_seq_length(layer_idx)
            if (
                max_length is not None
                and previous_seq_length + new_seq_length > max_length
            ):
                return max_length - new_seq_length
            return previous_seq_length

        Cache.get_usable_length = get_usable_length

    return {
        "cache_class": f"{Cache.__module__}.{Cache.__name__}",
        "get_usable_length": hasattr(Cache, "get_usable_length"),
        "get_max_length": hasattr(Cache, "get_max_length"),
        "semantics": "Transformers-4.45 cache length aliases only",
    }


def _processor(source, snapshot: Path):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(
        str(snapshot), min_pixels=source.H14X_MIN_PIXELS,
        max_pixels=source.H14X_MAX_PIXELS, use_fast=False,
        local_files_only=True,
    )


def load(method: str, dtype_name: str, snapshot: Path, scratch: Path,
         phase: str) -> Runtime:
    import torch

    if method not in NATIVE_METHODS | HOST_PORTS:
        raise ValueError(method)
    if dtype_name not in {"float16", "float32"}:
        raise ValueError(dtype_name)
    stack = environment_gate(phase)
    source_method = method if method in NATIVE_METHODS else "ours"
    source = h259_bridge.load_frozen_source(
        Path(__file__).resolve().parents[3], source_method,
        snapshot.parent.parent.parent if snapshot.name == MODEL_REVISION else snapshot.parent,
        scratch,
    )
    transformers_source = source.materialize_vendored_transformers_modeling()
    processor = _processor(source, snapshot)
    dtype = getattr(torch, dtype_name)
    receipt = {
        "stack": stack, "source_sha256": source.H259_SOURCE_SHA256,
        "rebound_source_sha256": source.H259_REBOUND_SHA256,
        "transformers_source": transformers_source,
        "snapshot": str(snapshot.resolve()), "model_revision": MODEL_REVISION,
        "attention_backend": "eager",
    }

    if method in HOST_PORTS:
        receipt["peer_migration_semantics"] = {
            "target": "Qwen2-VL vision-only migration of author CLIP method",
            "dtype_mode": dtype_name + "_storage_fp32_routing_and_reduction",
            "host_adaptations": "per-segment routing, representative RoPE, restore slots before PatchMerger",
            "dtome": "author score-ranked/capped selection; proportional attention before softmax",
            "pitome": "author post-block unweighted mean; existing stable ties and zero-removal guard",
            "independent_group_sums": "stable destination grouping and torch.segment_reduce",
        }

    if dtype_name == "float16" and method != "ipcv":
        receipt["fp16_eager_score_compat"] = (
            fp16_eager_stability.install_official_text_scores()
        )

    if method in {"ours", "dtome", "pitome"}:
        if method == "ours":
            receipt["adapter_source"] = source.materialize_adapter_module()
        from transformers import Qwen2VLForConditionalGeneration
        model, loading = Qwen2VLForConditionalGeneration.from_pretrained(
            str(snapshot), dtype=dtype,
            attn_implementation="eager", output_loading_info=True,
            local_files_only=True,
        )
        receipt["loading_info"] = _loading_summary(loading)
    elif method == "ipcv":
        receipt["official_source"] = source.materialize_official_source()
        if dtype_name == "float16":
            receipt["fp16_eager_score_compat"] = (
                fp16_eager_stability.patch_ipcv_text_scores(
                    source.OFFICIAL_ROOT / "Qwen2VL_IPCV" / "modeling_qwen2_vl_IPCV.py"
                )
            )
        receipt["ipcv_cache_api_compat"] = ensure_ipcv_legacy_cache_api()
        from Qwen2VL_IPCV import Qwen2VLConfig, Qwen2VLVisionConfig, get_model_class
        sparse = source.sparse_config_for("dense")
        vision = Qwen2VLVisionConfig.from_pretrained(
            str(snapshot), Sparse_config=sparse, local_files_only=True,
        )
        config = Qwen2VLConfig.from_pretrained(
            str(snapshot), Sparse_config=sparse, vision_config=vision,
            local_files_only=True,
        )
        receipt["ipcv_vision_initializer_compat"] = (
            ensure_ipcv_vision_initializer(config)
        )
        author_model_class = get_model_class("dart")
        model_class = ipcv_generation_compatible_class(author_model_class)
        receipt["ipcv_generation_compat"] = {
            "author_class": author_model_class.__name__,
            "runtime_class": model_class.__name__,
            "generation_mixin_added": model_class is not author_model_class,
        }
        model, loading = model_class.from_pretrained(
            str(snapshot), config=config, dtype=dtype,
            attn_implementation="eager",
            output_loading_info=True, local_files_only=True,
        )
        receipt["loading_info"] = _loading_summary(loading)
        receipt["official_config_model_type"] = model.config.model_type
    else:
        receipt["adapter_source"] = source.materialize_adapter_module()
        # Execute the pinned native Qwen block; retain only the common-stack
        # generation/placeholder interface around that author implementation.
        import hashlib
        import illava_native_adapter as multi_image_adapter
        sys.modules["illava_qwen2vl_adapter"] = multi_image_adapter
        extension_path = Path(multi_image_adapter.__file__)
        receipt["multi_image_extension"] = {
            "path": str(extension_path),
            "sha256": hashlib.sha256(extension_path.read_bytes()).hexdigest(),
            "scope": "pinned native Qwen selection; common-stack placeholder interface",
            "native_source": multi_image_adapter.SOURCE_RECEIPT,
            "dtype_mode": multi_image_adapter.dtype_mode(dtype),
        }
        from illava_qwen2vl_adapter import ILLavaQwen2VLForConditionalGeneration
        model, loading = ILLavaQwen2VLForConditionalGeneration.from_pretrained(
            str(snapshot), dtype=dtype,
            attn_implementation="eager", output_loading_info=True,
            local_files_only=True,
        )
        receipt["loading_info"] = _loading_summary(loading)

    # Match the SmolVLM formal loader: materialize on the host, then move the
    # complete single-GPU model as one unit.  This avoids Accelerate device-map
    # dispatch becoming an unrecorded execution difference between models.
    model = model.to("cuda").eval()
    model_backend = getattr(model.config, "_attn_implementation", None)
    visual_backend = getattr(getattr(model.visual, "config", None),
                             "_attn_implementation", model_backend)
    attention_classes = sorted({
        type(block.attn).__name__ for block in model.visual.blocks
    })
    if model_backend != "eager" or visual_backend != "eager":
        raise RuntimeError(
            f"actual attention backend differs from SmolVLM eager contract: "
            f"model={model_backend}, visual={visual_backend}"
        )
    if any("sdpa" in name.lower() or "flash" in name.lower()
           for name in attention_classes):
        raise RuntimeError(f"non-eager visual attention class loaded: {attention_classes}")
    receipt["actual_attention"] = {
        "model_config": model_backend,
        "visual_config": visual_backend,
        "visual_attention_classes": attention_classes,
    }
    if model.visual.get_dtype() != dtype:
        raise RuntimeError("visual tower precision differs from requested dtype")
    if len(model.visual.blocks) != 32 or not all(
        hasattr(block.attn, "qkv") for block in model.visual.blocks
    ):
        raise RuntimeError("Qwen fused-QKV visual contract changed")
    if method == "ours":
        h259_bridge.assert_ours_fused_contract(
            source, tuple(grid.H259_MODE[method][mode] for mode in grid.CONFIGS[method])
        )
    return Runtime(method, dtype_name, source, model, processor, receipt)


def source_mode(method: str, mode: str) -> str:
    if mode == "dense":
        return "dense"
    if method not in NATIVE_METHODS:
        return mode
    return grid.H259_MODE[method][mode]


def peer_config(method: str, mode: str) -> PeerConfig | None:
    if mode == "dense":
        return None
    value = grid.CONFIGS[method][mode]
    return PeerConfig(**value)


def timed_visual(runtime: Runtime, inputs: dict, mode: str):
    import torch

    captures = []
    handles = []
    illava_module = None
    original_illava_visual = None

    def begin(_module=None, _arguments=None):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        handles.append((start, end))

    def finish(_module, _arguments, output):
        if not handles:
            raise RuntimeError("visual timing end observed without a start")
        start, end = handles[-1]
        end.record()
        torch.cuda.synchronize()
        captures.append((output.detach().clone(), float(start.elapsed_time(end)), None))

    # The official iLLaVA port manually traverses the vision blocks so its
    # active path intentionally bypasses visual.forward.  Time that complete
    # visual function; every other method uses the same model.visual hooks as
    # the formal SmolVLM runner.
    if runtime.method == "illava":
        illava_module = importlib.import_module("illava_qwen2vl_adapter")
        original_illava_visual = illava_module.run_illava_visual

        def measured_illava_visual(*args, **kwargs):
            begin()
            output, profile = original_illava_visual(*args, **kwargs)
            start, end = handles[-1]
            end.record()
            torch.cuda.synchronize()
            captures.append((
                output.detach().clone(), float(start.elapsed_time(end)), profile,
            ))
            return output, profile

        illava_module.run_illava_visual = measured_illava_visual
        pre = post = None
        timing_boundary = "generate_inner_complete_illava_visual"
    else:
        pre = runtime.model.visual.register_forward_pre_hook(begin)
        post = runtime.model.visual.register_forward_hook(finish)
        timing_boundary = "generate_inner_model_visual_hook"
    try:
        with online_context(runtime, inputs, mode) as active, torch.inference_mode():
            generated = runtime.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=1,
                use_cache=True,
            )
            torch.cuda.synchronize()
            if len(captures) != 1:
                raise RuntimeError(
                    f"expected exactly one complete visual execution, got {len(captures)}"
                )
            features, elapsed, captured_profile = captures[0]
            profile = online_profile(
                runtime, inputs, mode, active, captured_profile, timing_boundary
            )
        del generated
        return features, elapsed, profile
    finally:
        if pre is not None:
            pre.remove()
        if post is not None:
            post.remove()
        if illava_module is not None:
            illava_module.run_illava_visual = original_illava_visual


def online_profile(runtime: Runtime, inputs: dict, mode: str, active,
                   captured_profile: dict | None, timing_boundary: str) -> dict:
    if runtime.method == "ours" and active is not None:
        profile = active.summary()
    elif runtime.method == "ipcv":
        profile = {
            "config": active,
            "input_tokens": int(inputs["pixel_values"].shape[0]),
        }
    elif runtime.method == "illava":
        profile = captured_profile
    elif active is not None:
        native = int(inputs["image_grid_thw"].prod(dim=1).sum().item())
        depth = len(active.trace)
        profile = {
            "path_active": bool(active.path_active),
            "compression_active": bool(active.active),
            "native_tokens": native,
            "restored_tokens": int(active.restored_tokens),
            "attention_work_ratio": sum(
                row["attention_tokens"] for row in active.trace
            ) / (depth * native),
            "mlp_work_ratio": sum(
                row["mlp_tokens"] for row in active.trace
            ) / (depth * native),
            "route_digest": active.routing_digest(),
            "trace": active.trace,
        }
    else:
        profile = {
            "path_active": False,
            "compression_active": False,
            "attention_work_ratio": 1.0,
            "mlp_work_ratio": 1.0,
            "route_digest": None,
        }
    profile = dict(profile or {})
    profile["timing_boundary"] = timing_boundary
    profile["timed_via_generate"] = True
    return profile


@contextlib.contextmanager
def online_context(runtime: Runtime, inputs: dict, mode: str):
    if runtime.method in NATIVE_METHODS:
        adapter = h259_bridge.set_generation_mode(
            runtime.source, runtime.model, inputs,
            source_mode(runtime.method, mode),
        )
        try:
            if runtime.method == "ipcv":
                yield runtime.source.sparse_config_for(
                    source_mode(runtime.method, mode)
                )
            else:
                yield adapter
        finally:
            if adapter is not None:
                adapter.close()
        return
    port = QwenPeerVisionPort(runtime.model.visual, peer_config(runtime.method, mode)) \
        if mode != "dense" else None
    try:
        if port is not None:
            port.__enter__()
        yield port
    finally:
        if port is not None:
            port.close()


