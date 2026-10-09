# Publication copy: comments/line endings only; algorithm AST unchanged.
# Project migration copy; retain upstream method and host notices. No experiment executed.
# Component licensing and exact source hashes: sources/FILE_DISPOSITIONS.json.
"""Single-GPU ordinary LLaVA runtime for five H297 method operations."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.metadata
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from grid import JOB_SPECS, MODEL_ID, MODEL_REVISION
from prompting import PITOME_IMAGE_TOKENS, legacy_llava_prompt, rewrite_image_runs

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "frozen_sources"
OURS_FROZEN = FROZEN / "ours_h297"
OURS_FP32_PORT = FROZEN / "ours_fp32_port"
OURS_MODULES = (
    "layout.py", "adapter_h253_reference.py", "relation_triton.py",
    "adapter_triton_relation.py", "pack_restore_triton.py",
    "adapter_pack_restore_triton.py", "adapter_triton_eager.py",
    "adapter_pack_restore_eager.py", "attention_restore_triton.py",
    "adapter_q_fused_eager.py",
)


def environment_gate(job: int, *, phase: str = "formal") -> dict:
    import torch

    from check_environment import check_environment
    locked = check_environment(HERE)
    method, dtype_name, target_gpu = JOB_SPECS[job]
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("SmolVLM formal stack requires Python 3.11")
    if torch.__version__ != "2.5.1+cu121":
        raise RuntimeError(f"SmolVLM formal PyTorch mismatch: {torch.__version__}")
    version = importlib.metadata.version("transformers")
    if version != "4.57.6":
        raise RuntimeError(f"SmolVLM formal Transformers mismatch: {version}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("H303 requires exactly one visible GPU")
    gpu = torch.cuda.get_device_name(0)
    minimum_gib = 30 if target_gpu == "V100" else 70
    memory = torch.cuda.get_device_properties(0).total_memory
    if target_gpu not in gpu or memory < minimum_gib * 1024**3:
        raise RuntimeError(f"job {job} requires {target_gpu} >= {minimum_gib} GiB, got {gpu}")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return {
        "packages": locked["packages"], "environment_lock_sha256": locked["environment_lock_sha256"],
        "job": job, "method": method, "dtype": dtype_name,
        "gpu": gpu, "gpu_memory_bytes": memory, "target_gpu": target_gpu,
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda": torch.version.cuda, "transformers": version,
        "attention": "eager", "tf32": False, "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def frozen_source_hashes() -> dict[str, str]:
    names = ("clip_peer_ports.py", "peer_port_core.py", "pitome_author_fp16.py",
             "run_port_job.py", "llava_vision_loader.py",
             "author_dymu_core.py", "author_dymu_bridge.py", "ipcv_clip_batch.py")
    peers = {name: hashlib.sha256((FROZEN / name).read_bytes()).hexdigest() for name in names}
    ours = {f"ours_h297/{name}": hashlib.sha256((OURS_FROZEN / name).read_bytes()).hexdigest()
            for name in OURS_MODULES}
    port = {f"ours_fp32_port/{name}": hashlib.sha256(
        (OURS_FP32_PORT / name).read_bytes()).hexdigest()
        for name in ("relation_triton.py", "pack_restore_triton.py")}
    return peers | ours | port


def resolve_snapshot(snapshot: Path) -> Path:
    """Use the supplied offline cache path without consulting the default Hub cache."""
    pinned = Path(snapshot).resolve()
    model_cache = f"models--{MODEL_ID.replace('/', '--')}"
    if (pinned.name != MODEL_REVISION or pinned.parent.name != "snapshots"
            or pinned.parent.parent.name != model_cache):
        raise ValueError(f"not a pinned LLaVA cache path: {pinned}")
    if not (pinned / "config.json").is_file():
        raise FileNotFoundError(pinned / "config.json")
    return pinned


@dataclass
class Runtime:
    job: int
    method: str
    dtype_name: str
    model: object
    processor: object
    dense_tower: object
    candidate_bridge: object
    receipt: dict
    native_dense_tower: object = None

    @property
    def owner(self):
        return self.model.model

    @contextlib.contextmanager
    def mode(self, mode: str):
        if mode not in ("dense", "candidate", "canonical_dense", "native_dense_diagnostic"):
            raise ValueError(mode)
        if mode == "candidate":
            tower = self.candidate_bridge
        elif mode == "native_dense_diagnostic" and self.native_dense_tower is not None:
            tower = self.native_dense_tower
        else:
            tower = self.dense_tower
        # SmolVLM installs the method before the native vision hooks start.
        with contextlib.ExitStack() as contexts:
            if mode == "candidate" and self.method == "ours":
                contexts.enter_context(self.candidate_bridge.managed_adapter)
            self.owner.vision_tower = tower
            try:
                yield
            finally:
                self.owner.vision_tower = self.dense_tower

    def prepare(self, row: dict, data_root: Path, mode: str) -> dict:
        import torch
        from PIL import Image

        images = []
        try:
            for value in row["images"]:
                path = (data_root / value).resolve()
                if not path.is_relative_to(data_root.resolve()) or not path.is_file():
                    raise RuntimeError(f"missing/outside LLaVA image: {path}")
                with Image.open(path) as opened:
                    images.append(opened.convert("RGB"))
            prompt = legacy_llava_prompt(row["messages"])
            if prompt.count("<image>") != len(images):
                raise RuntimeError("LLaVA prompt/image count mismatch")
            values = self.processor(text=prompt, images=images, return_tensors="pt")
            if mode == "candidate" and self.method == "pitome":
                ids = rewrite_image_runs(
                    values["input_ids"][0].tolist(), self.model.config.image_token_id,
                    [PITOME_IMAGE_TOKENS] * len(images),
                )
                values["input_ids"] = torch.tensor([ids], dtype=torch.long)
                values["attention_mask"] = torch.ones_like(values["input_ids"])
            dtype = getattr(torch, self.dtype_name)
            return {
                key: (value.to("cuda:0", dtype=dtype) if value.is_floating_point()
                      else value.to("cuda:0"))
                for key, value in values.items()
            }
        finally:
            for image in images:
                image.close()

    def timed_visual(self, inputs: dict, mode: str):
        import torch

        with self.mode(mode), torch.inference_mode():
            tower = self.owner.vision_tower
            events, walls, features = [], [], []

            def before(_module, _arguments):
                torch.cuda.synchronize()
                pair = (torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True))
                events.append(pair)
                walls.append([time.perf_counter(), None])
                pair[0].record()

            def after(_module, _arguments, output):
                events[-1][1].record()
                torch.cuda.synchronize()
                walls[-1][1] = time.perf_counter()
                features.append(output.hidden_states[-2][:, 1:])

            pre = tower.register_forward_pre_hook(before)
            post = tower.register_forward_hook(after)
            try:
                torch.cuda.synchronize()
                tick = time.perf_counter()
                generated = self.model.generate(
                    **inputs, do_sample=False, num_beams=1,
                    max_new_tokens=1, use_cache=True,
                )
                torch.cuda.synchronize()
                ttft_ms = (time.perf_counter() - tick) * 1000
                if len(events) != 1 or len(walls) != 1 or len(features) != 1:
                    raise RuntimeError("expected exactly one LLaVA vision forward")
                if generated.shape[1] - inputs["input_ids"].shape[1] != 1:
                    raise RuntimeError("timing generate did not decode one token")
                feature = features[0].detach().clone()
                measures = {
                    "vision_wall_ms": (walls[0][1] - walls[0][0]) * 1000,
                    "vision_device_elapsed_ms": float(events[0][0].elapsed_time(events[0][1])),
                    "ttft_wall_ms": ttft_ms,
                    "new_tokens": 1,
                }
            finally:
                pre.remove()
                post.remove()
        profile = ({"active": False} if mode in ("dense", "canonical_dense", "native_dense_diagnostic") else self.candidate_bridge.receipt)
        return feature, measures, profile

    def generate(self, inputs: dict, mode: str, *, max_new_tokens: int = 2048) -> dict:
        import torch

        with self.mode(mode), torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, num_beams=1, min_new_tokens=0,
                max_new_tokens=max_new_tokens, use_cache=True,
            )
            torch.cuda.synchronize()
        new = output[:, inputs["input_ids"].shape[1]:]
        ids = [int(value) for value in new[0].detach().cpu().tolist()]
        answer = self.processor.batch_decode(
            new, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        profile = None if mode in ("dense", "canonical_dense", "native_dense_diagnostic") else self.candidate_bridge.receipt
        return {"raw_text": answer, "token_ids": ids, "generated_tokens": len(ids),
                "hit_cap": len(ids) >= max_new_tokens, "visual_profile": profile}


    def diagnose_native_dense(self, inputs: dict, *, max_new_tokens: int = 16) -> dict:
        """Explicit small-input diagnostic; never run as part of paired timing.

        Call after load() with a representative already-prepared request. This
        observes conversion/backend effects and does not change pass tolerances.
        """
        import torch
        if self.method != "dtome" or self.native_dense_tower is None:
            raise ValueError("native dense diagnostic applies only to DToMe")
        canonical, _, _ = self.timed_visual(inputs, "canonical_dense")
        native, _, _ = self.timed_visual(inputs, "native_dense_diagnostic")
        limits = {"float32": (2e-5, 2e-6), "float16": (5e-3, 5e-3)}
        rtol, atol = limits[self.dtype_name]
        delta = native.float() - canonical.float()
        canonical_text = self.generate(inputs, "dense", max_new_tokens=max_new_tokens)
        native_text = self.generate(inputs, "native_dense_diagnostic", max_new_tokens=max_new_tokens)
        return {
            "feature_shape_equal": native.shape == canonical.shape,
            "feature_close": bool(torch.allclose(native, canonical, rtol=rtol, atol=atol)),
            "feature_max_abs": float(delta.abs().max().item()),
            "feature_relative_l2": float(delta.norm().item() / max(canonical.float().norm().item(), 1e-12)),
            "rtol": rtol, "atol": atol, "max_new_tokens": max_new_tokens,
            "generation_tokens_equal": canonical_text["token_ids"] == native_text["token_ids"],
            "canonical": canonical_text, "native_disabled_merge": native_text,
            "native_dispatch": self.native_dense_tower.runner.backend_observation,
            "interpretation": "descriptive native-disabled versus canonical HF observation; not an author-fidelity gate",
        }


def load(job: int, snapshot: Path, scratch: Path, *, phase: str = "formal") -> Runtime:
    import torch
    import torch.nn as nn
    from transformers import (AutoTokenizer, CLIPImageProcessor, CLIPVisionModel,
                              LlavaForConditionalGeneration, LlavaProcessor)
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    snapshot = resolve_snapshot(snapshot)
    stack = environment_gate(job, phase=phase)
    method, dtype_name, _ = JOB_SPECS[job]
    source_hashes = frozen_source_hashes()
    sys.path.insert(0, str(FROZEN))
    from clip_peer_ports import ClipPeerRunner
    from run_port_job import config_for

    ours_adapter = None
    if method == "ours":
        sys.path.insert(0, str(OURS_FROZEN))
        sys.path.insert(0, str(OURS_FP32_PORT))
        from adapter_q_fused_eager import CompleteManagedOperatorDelegationAdapter
        from layout import build_layout
        import relation_triton
        import pack_restore_triton
        if Path(relation_triton.__file__).resolve() != (OURS_FP32_PORT / "relation_triton.py").resolve():
            raise RuntimeError("FP32 relation port was not imported")
        if Path(pack_restore_triton.__file__).resolve() != (
                OURS_FP32_PORT / "pack_restore_triton.py").resolve():
            raise RuntimeError("FP32 Triton mask port was not imported")

    image_processor = CLIPImageProcessor.from_pretrained(snapshot, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
    processor = LlavaProcessor(
        image_processor=image_processor, tokenizer=tokenizer, patch_size=14,
        num_additional_image_tokens=1, vision_feature_select_strategy="default",
    )
    dtype = getattr(torch, dtype_name)
    model, loading = LlavaForConditionalGeneration.from_pretrained(
        snapshot, dtype=dtype, low_cpu_mem_usage=True,
        attn_implementation="eager", local_files_only=True,
        output_loading_info=True,
    )
    load_keys = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    if any(loading.get(key) for key in load_keys):
        raise RuntimeError(f"LLaVA checkpoint mismatch: {loading}")
    model = model.to("cuda:0").eval()
    owner = model.model
    dense_tower = owner.vision_tower
    if getattr(dense_tower.config, "_attn_implementation", None) != "eager":
        raise RuntimeError("LLaVA CLIP attention is not eager")
    candidate = CLIPVisionModel(copy.deepcopy(dense_tower.config))
    incompatible = candidate.load_state_dict(dense_tower.state_dict(), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("candidate CLIP checkpoint mismatch")
    candidate = candidate.to("cuda:0", dtype=dtype).eval()

    if method == "ours":
        from grid import METHOD_POINTS
        point = METHOD_POINTS["ours"]
        holder = SimpleNamespace(model=SimpleNamespace(vision_tower=candidate),
                                 config=SimpleNamespace(vision_feature_layer=-2))
        ours_adapter = CompleteManagedOperatorDelegationAdapter(
            holder, layout=build_layout(24, point["group_side"], point["targets_per_group"]),
            mlp_threshold=point["mlp_threshold"], q_threshold=point["q_threshold"],
            mlp_start_layer=point["mlp_start_layer"], q_start_layer=point["q_start_layer"],
            eligibility_mode=point["eligibility_mode"],
            cross_layer_threshold=point["cross_layer_threshold"],
        )

    pitome_patch = None
    if method == "pitome":
        scratch.mkdir(parents=True, exist_ok=True)
        os.environ["H297_OUTPUT"] = str(scratch)
        import pitome_author_fp16 as pitome_source
        pitome_source.materialize_author_sources()
        sys.path.insert(0, str(pitome_source.AUTHOR_ROOT.parent))
        import author_pitome.patch.clip_hf as pitome_patch
        pitome_patch.apply_patch(candidate.vision_model.encoder)
        candidate.vision_model.encoder.ratio = 0.96

    dtome_runner = None
    if method == "dtome":
        from author_dymu_bridge import AuthorDToMeRunner
        dtome_runner = AuthorDToMeRunner(candidate, config_for(method))

    ipcv_runner = None
    if method == "ipcv":
        from ipcv_clip_batch import ClipIPCVBatchRunner
        ipcv_runner = ClipIPCVBatchRunner(candidate, config_for(method))

    class Bridge(nn.Module):
        def __init__(self, tower, managed_adapter=None):
            super().__init__()
            self.tower = tower
            self.managed_adapter = managed_adapter
            self.dtome_runner = dtome_runner
            self.ipcv_runner = ipcv_runner
            self.receipt = None

        @property
        def config(self):
            return self.tower.config

        @property
        def device(self):
            return next(self.tower.parameters()).device

        @property
        def dtype(self):
            return next(self.tower.parameters()).dtype

        def forward(self, pixel_values, *args, **kwargs):
            if self.dtome_runner is not None:
                # Keep the image batch intact: the author chooses B=1 threshold
                # matching or batch-level allocation/padding itself.
                patches = self.dtome_runner(pixel_values)
                if patches.shape[1] != 576 or self.dtome_runner.restored_tokens != 577:
                    raise RuntimeError("author DToMe did not restore native LLaVA patch slots")
                if not self.dtome_runner.active:
                    raise RuntimeError("inactive author DToMe visual path")
                full = torch.cat((patches.new_zeros((patches.shape[0], 1, patches.shape[-1])),
                                  patches), dim=1)
                self.receipt = {
                    "method": method, "images": patches.shape[0], "active": True,
                    "scope": "author-native visual DToMe; host slot restoration; no LLM VTU",
                    "author_commit": "3770d6add8fd9b972ae611f3f3b16a552dea13cc",
                    "restored_tokens": self.dtome_runner.restored_tokens,
                    "trace": self.dtome_runner.trace,
                    "weight_conversion": self.dtome_runner.conversion,
                    "native_dispatch": self.dtome_runner.backend_observation,
                    "per_image": [{
                        "active": any(row["output_tokens_per_image"][i] <
                                      row["input_tokens_per_image"][i]
                                      for row in self.dtome_runner.trace),
                        "selected_visual_tokens": 576, "restored_tokens": 577,
                        "trace": [{"layer_zero_based": row["layer_zero_based"],
                                   "input_tokens": row["input_tokens_per_image"][i],
                                   "output_tokens": row["output_tokens_per_image"][i]}
                                  for row in self.dtome_runner.trace],
                    } for i in range(patches.shape[0])],
                }
                return BaseModelOutputWithPooling(
                    last_hidden_state=full, pooler_output=None,
                    hidden_states=(full, full), attentions=None,
                )
            if self.ipcv_runner is not None:
                patches = self.ipcv_runner(pixel_values)
                if patches.shape[1] != 576 or self.ipcv_runner.restored_tokens != 577:
                    raise RuntimeError("batched IPCV did not restore native LLaVA patch slots")
                if not self.ipcv_runner.active:
                    raise RuntimeError("inactive batched IPCV visual path")
                full = torch.cat((patches.new_zeros((patches.shape[0], 1, patches.shape[-1])),
                                  patches), dim=1)
                self.receipt = {
                    "method": method, "images": patches.shape[0], "active": True,
                    "scope": "InternVL IPCV logical CLIP migration; per-image selection; global kNN including CLS",
                    "restored_tokens": self.ipcv_runner.restored_tokens,
                    "trace": self.ipcv_runner.trace,
                    "per_image": [{"active": True, "selected_visual_tokens": 576,
                                   "restored_tokens": 577, "trace": self.ipcv_runner.trace}
                                  for _ in range(patches.shape[0])],
                }
                return BaseModelOutputWithPooling(
                    last_hidden_state=full, pooler_output=None,
                    hidden_states=(full, full), attentions=None,
                )
            full_images, traces = [], []
            for pixels in pixel_values.split(1, dim=0):
                if method == "ours":
                    if not self.managed_adapter._entered:
                        raise RuntimeError("Ours must be installed by Runtime.mode before vision timing")
                    result = self.tower(pixels, output_hidden_states=True, return_dict=True)
                    # The frozen first encoder layer resets state for each image.
                    work = self.managed_adapter.summary()
                    full = result.hidden_states[-2]
                    if full.shape[1] != 577 or not work["kv_full"]:
                        raise RuntimeError("Ours did not restore native LLaVA patch slots")
                    active = work["packed_mlp_calls"] > 0 and work["hidden_state_restore_calls"] > 0
                    traces.append({"active": active, "selected_visual_tokens": 576,
                                   "restored_tokens": 577, "work": work})
                elif method == "pitome":
                    result = self.tower(pixels, output_hidden_states=True, return_dict=True)
                    full = result.hidden_states[-2]
                    trace = [int(value.shape[1]) for value in result.hidden_states]
                    if full.shape[1] - 1 != PITOME_IMAGE_TOKENS:
                        raise RuntimeError("PiToMe selected feature count changed")
                    traces.append({"active": min(trace) < 577, "min_internal_tokens": min(trace),
                                   "selected_visual_tokens": int(full.shape[1] - 1), "trace": trace})
                else:
                    runner = ClipPeerRunner(self.tower, config_for(method))
                    patches = runner(pixels)
                    if patches.shape[1] != 576 or runner.restored_tokens != 577:
                        raise RuntimeError(f"{method} did not restore native LLaVA patch slots")
                    full = torch.cat((patches.new_zeros((1, 1, patches.shape[-1])), patches), dim=1)
                    traces.append({"active": bool(runner.active),
                                   "selected_visual_tokens": int(patches.shape[1]),
                                   "restored_tokens": int(runner.restored_tokens),
                                   "trace": runner.trace})
                full_images.append(full)
            if not traces or not all(row["active"] for row in traces):
                raise RuntimeError(f"inactive {method} visual path")
            full = torch.cat(full_images, dim=0)
            self.receipt = {"method": method, "images": len(traces), "per_image": traces,
                            "active": True}
            return BaseModelOutputWithPooling(
                last_hidden_state=full, pooler_output=None,
                hidden_states=(full, full), attentions=None,
            )

    bridge = Bridge(candidate, ours_adapter).eval()
    native_dense_tower = None
    if method == "dtome":
        class NativeDenseBridge(nn.Module):
            def __init__(self):
                super().__init__()
                self.runner = AuthorDToMeRunner(candidate, config_for(method), merge_enabled=False)
                self.config = candidate.config

            @property
            def device(self):
                return next(self.runner.parameters()).device

            @property
            def dtype(self):
                return next(self.runner.parameters()).dtype

            def forward(self, pixel_values, *args, **kwargs):
                patches = self.runner(pixel_values)
                full = torch.cat((patches.new_zeros((patches.shape[0], 1, patches.shape[-1])),
                                  patches), dim=1)
                return BaseModelOutputWithPooling(
                    last_hidden_state=full, pooler_output=None,
                    hidden_states=(full, full), attentions=None,
                )

        native_dense_tower = NativeDenseBridge().eval()
    receipt = {"stack": stack, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
               "snapshot": str(snapshot), "frozen_h297_sources": source_hashes,
               "tokenizer_class": type(tokenizer).__name__, "tokenizer_use_fast": True,
               "assistant_suffix_trailing_space": False,
               "method_context": "installed before native vision timing, restored after generate",
               "loading_info": {key: loading.get(key, []) for key in load_keys},
               "candidate_vision_attention": (
                   "author torch.nn.MultiheadAttention need_weights=False; existing PyTorch dispatch setting unchanged"
                   if method == "dtome" else "HF CLIP eager"),
               "primary_dense_vision_attention": "canonical HF CLIP eager (original r7 timing contract)",
               "native_disabled_diagnostic_attention": (
                   "author MultiheadAttention r_total=0; existing dispatch setting unchanged; diagnostic only"
                   if method == "dtome" else None),
               "native_mha_fastpath_enabled_at_load": (
                   bool(torch.backends.mha.get_fastpath_enabled()) if method == "dtome" else None),
               "canonical_dense_and_llm_attention": "HF eager",
               "dense_reference_contract": (
                   "primary dense timing and quality are canonical HF eager, as declared in original r7; "
                   "native_dense_diagnostic is separate and never the primary denominator; "
                   "native candidate dispatch may differ, so the ratio is a complete-path comparison, not pure merging acceleration"
                   if method == "dtome" else "canonical HF dense"),
               "bridge": ("author-native batched visual DToMe; original image slots restored"
                          if method == "dtome" else
                          "per-image IPCV selection; batch-global neighbors including CLS"
                          if method == "ipcv" else
                          "per-image H297 operation; original image order retained"),
               "ours_relation_port": (str(OURS_FP32_PORT / "relation_triton.py")
                                      if method == "ours" else None),
               "ours_pack_port": (str(OURS_FP32_PORT / "pack_restore_triton.py")
                                  if method == "ours" else None)}
    return Runtime(job, method, dtype_name, model, processor, dense_tower, bridge, receipt,
                   native_dense_tower=native_dense_tower)
