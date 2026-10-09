# Publication copy: comments/line endings only; algorithm AST unchanged.
# Project migration copy; retain upstream method and host notices. No experiment executed.
# Component licensing and exact source hashes: sources/FILE_DISPOSITIONS.json.
"""Runtime interface for the unchanged SmolVLM quality/timing/journal engine."""
from __future__ import annotations
import contextlib
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from dataclasses import dataclass

import grid
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from onevision_ports import PeerBridge, frozen_source_hashes as peer_hashes, features_equal


def frozen_source_hashes():
    paths = ('onevision_ports.py', 'environment_lock.json', 'precision_gate_rope_r1.py')
    return peer_hashes() | {name: hashlib.sha256((HERE.parent / name).read_bytes()).hexdigest()
                            for name in paths}


def resolve_snapshot(snapshot):
    snapshot = Path(snapshot).resolve()
    if not (snapshot / 'config.json').is_file():
        raise FileNotFoundError(snapshot / 'config.json')
    if grid.TARGET_DTYPE == 'float16':
        model_cache = f'models--{grid.MODEL_ID.replace("/", "--")}'
        if (snapshot.name != grid.MODEL_REVISION or snapshot.parent.name != 'snapshots'
                or snapshot.parent.parent.name != model_cache):
            raise ValueError('not the pinned OneVision HF snapshot')
    elif not (snapshot / 'CONVERSION_RECEIPT.json').is_file():
        raise ValueError('FP32 needs the accepted original-BF16 conversion receipt')
    return snapshot


def environment_gate(job, *, phase='formal'):
    import torch
    lock = json.loads((HERE.parent / 'environment_lock.json').read_text())
    versions = {name: importlib.metadata.version(name) for name in lock['packages']}
    if sys.version.split()[0] != lock['python'] or versions != lock['packages']:
        raise RuntimeError('exact SmolVLM environment mismatch')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('formal measurement requires one visible GPU')
    gpu = torch.cuda.get_device_name(0)
    if grid.TARGET_GPU not in gpu:
        raise RuntimeError(f'expected {grid.TARGET_GPU}, got {gpu}')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return {'job': job, 'method': grid.JOB_SPECS[job][0], 'dtype': grid.TARGET_DTYPE,
            'gpu': gpu, 'python': lock['python'], 'torch': versions['torch'],
            'transformers': versions['transformers'], 'packages': versions,
            'cuda': torch.version.cuda, 'attention': 'eager', 'language_attention': grid.TEXT_ATTENTION,
            'tf32': False, 'threads': 2}


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
    phase: str = 'formal'

    @property
    def owner(self):
        return self.model.model

    @contextlib.contextmanager
    def mode(self, mode):
        if mode not in ('dense', 'candidate'):
            raise ValueError(mode)
        self.owner.vision_tower = self.dense_tower if mode == 'dense' else self.candidate_bridge
        try:
            yield
        finally:
            self.owner.vision_tower = self.dense_tower

    def prepare(self, row, data_root, mode):
        import torch
        from PIL import Image
        images = []
        try:
            for value in row['images']:
                path = (data_root / value).resolve()
                if not path.is_relative_to(data_root.resolve()) or not path.is_file():
                    raise RuntimeError(f'missing or outside image: {path}')
                with Image.open(path) as opened:
                    images.append(opened.convert('RGB'))
            content = [{'type': item['type'], **({'text': item['value']} if item['type'] == 'text' else {})}
                       for item in row['messages']]
            if sum(item['type'] == 'image' for item in content) != len(images):
                raise RuntimeError('OneVision prompt/image mismatch')
            prompt = self.processor.apply_chat_template([{'role': 'user', 'content': content}],
                                                         tokenize=False, add_generation_prompt=True)
            values = self.processor(text=prompt, images=[images], return_tensors='pt')
            if values['batch_num_images'].tolist() != [len(images)]:
                raise RuntimeError('OneVision processor lost image count')
            return {key: value.to('cuda:0', dtype=getattr(torch, self.dtype_name)) if value.is_floating_point()
                    else value.to('cuda:0') for key, value in values.items()}
        finally:
            for image in images:
                image.close()

    def timed_visual(self, inputs, mode):
        import torch
        events, walls, features = [], [], []
        def before(_module, _arguments):
            torch.cuda.synchronize()
            pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events.append(pair)
            walls.append([time.perf_counter(), None])
            pair[0].record()
        def after(_module, _arguments, output):
            events[-1][1].record()
            torch.cuda.synchronize()
            walls[-1][1] = time.perf_counter()
            features.append(output.hidden_states[-1])
        with self.mode(mode), torch.inference_mode():
            tower = self.owner.vision_tower
            pre, post = tower.register_forward_pre_hook(before), tower.register_forward_hook(after)
            try:
                torch.cuda.synchronize()
                tick = time.perf_counter()
                output = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                             max_new_tokens=1, use_cache=True)
                torch.cuda.synchronize()
                ttft = (time.perf_counter() - tick) * 1000
                if len(events) != 1 or output.shape[1] - inputs['input_ids'].shape[1] != 1:
                    raise RuntimeError('expected one native tower forward and one decoded token')
                feature = features[0].detach().clone()
                measures = {'vision_wall_ms': (walls[0][1] - walls[0][0]) * 1000,
                            'vision_device_elapsed_ms': float(events[0][0].elapsed_time(events[0][1])),
                            'ttft_wall_ms': ttft, 'new_tokens': 1}
            finally:
                pre.remove()
                post.remove()
        profile = {'active': False} if mode == 'dense' else {
            **self.candidate_bridge.receipt, 'images': int(inputs['batch_num_images'][0])}
        if mode == 'candidate' and self.phase == 'smoke':
            profile['image_validation'] = 'skipped by user request'
        return feature, measures, profile

    def generate(self, inputs, mode, *, max_new_tokens=2048):
        import torch
        with self.mode(mode), torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                         min_new_tokens=0, max_new_tokens=max_new_tokens, use_cache=True)
            torch.cuda.synchronize()
        new = output[:, inputs['input_ids'].shape[1]:]
        ids = new[0].detach().cpu().tolist()
        text = self.processor.batch_decode(new, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0].strip()
        profile = None if mode == 'dense' else {
            **self.candidate_bridge.receipt, 'images': int(inputs['batch_num_images'][0])}
        if profile is not None and self.phase == 'smoke':
            profile['image_validation'] = 'skipped by user request'
        return {'raw_text': text, 'token_ids': ids, 'generated_tokens': len(ids),
                'hit_cap': len(ids) >= max_new_tokens,
                'visual_profile': profile}

    def disabled_gate(self, inputs):
        import torch
        disabled = PeerBridge(self.dense_tower, self.method,
                              config=grid.METHOD_POINTS[self.method], disabled=True)
        args = {key: inputs[key] for key in ('pixel_values', 'image_sizes', 'batch_num_images')}
        with torch.inference_mode():
            native = self.owner.get_image_features(**args)
            self.owner.vision_tower = disabled
            try:
                control = self.owner.get_image_features(**args)
                output = self.model.generate(**inputs, do_sample=False, max_new_tokens=2048, use_cache=True)
            finally:
                self.owner.vision_tower = self.dense_tower
            dense = self.model.generate(**inputs, do_sample=False, max_new_tokens=2048, use_cache=True)
            cleanup = self.owner.get_image_features(**args)
        checks = {'features_exact': features_equal(native, control, exact=True),
                  'tokens_exact': torch.equal(output, dense),
                  'cleanup_exact': features_equal(native, cleanup, exact=True)}
        if not all(checks.values()):
            raise RuntimeError(f'disabled Dense identity failed: {checks}')
        return checks


def load(job, snapshot, scratch, *, phase='formal'):
    import torch
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
    stack = environment_gate(job, phase=phase)
    snapshot = resolve_snapshot(snapshot)
    provenance = {'source_dtype': 'F16', 'runtime_dtype': grid.TARGET_DTYPE,
                  'model_id': grid.MODEL_ID, 'revision': grid.MODEL_REVISION}
    if grid.TARGET_DTYPE == 'float32':
        recipe_dir = HERE.parent / 'frozen_fp32_recipe'
        spec = importlib.util.spec_from_file_location('h304_fp32_recipe', recipe_dir / 'common.py')
        recipe = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(recipe_dir))
        spec.loader.exec_module(recipe)
        provenance = recipe.validate_assets(snapshot, recipe_dir / 'data/POPE.tsv')
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    dtype = getattr(torch, grid.TARGET_DTYPE)
    model, loading = LlavaOnevisionForConditionalGeneration.from_pretrained(
        snapshot, torch_dtype=dtype, low_cpu_mem_usage=True, local_files_only=True,
        attn_implementation={'vision_config': 'eager', 'text_config': grid.TEXT_ATTENTION}, output_loading_info=True)
    keys = ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')
    if any(loading.get(key) for key in keys):
        raise RuntimeError(f'OneVision strict loading failed: {loading}')
    model.to('cuda:0').eval()
    from precision_gate_rope_r1 import validate_model_precision
    precision_receipt = validate_model_precision(model, dtype, torch)
    if model.config.vision_feature_layer != -1 or model.config.vision_feature_select_strategy != 'full':
        raise RuntimeError('native OneVision feature selection changed')
    backends = {'vision': model.config.vision_config._attn_implementation,
                'text': model.config.text_config._attn_implementation}
    if backends != {'vision': 'eager', 'text': grid.TEXT_ATTENTION}:
        raise RuntimeError(f'native attention backend mismatch: {backends}')
    method = grid.JOB_SPECS[job][0]
    tower = model.model.vision_tower
    bridge = PeerBridge(tower, method, config=grid.METHOD_POINTS[method])
    receipt = {'stack': stack, 'model_id': grid.MODEL_ID, 'model_revision': grid.MODEL_REVISION,
               'loading_info': {key: loading.get(key, []) for key in keys},
               'frozen_smol_sources': frozen_source_hashes(), 'weight_provenance': provenance,
               'actual_backends': backends, 'parameter_buffer_precision': precision_receipt,
               'bridge': 'native OneVision DToMe/iLLaVA calls; logical PiToMe/Qwen-IPCV migration; 729-slot restoration; native packing',
               'candidate_tower_shares_dense_storage': True}
    return Runtime(job, method, grid.TARGET_DTYPE, model, processor, tower, bridge, receipt, phase)
