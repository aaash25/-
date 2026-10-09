# Publication copy: comments/line endings only; algorithm AST unchanged.
# Project migration copy; retain upstream method and host notices. No experiment executed.
# Component licensing and exact source hashes: sources/FILE_DISPOSITIONS.json.
"""H307 author-vision four-peer matrix; SmolVLM protocol is reused verbatim."""
from dataset_registry import DATASETS, DATASET_SIZES
MODEL_ID = 'lmms-lab/llava-onevision-qwen2-7b-ov'
MODEL_REVISION = '0b07bf7565e244cf4f39982249eafe8cd799d6dd'
EXPERIMENT = 'H307'
RELEASE = 'onevision_ipcv_global_float32_r1'
TARGET_GPU = 'A800'
TARGET_DTYPE = 'float32'
TEXT_ATTENTION = 'sdpa'  # Native H290 Qwen2 backend; visual protocol remains eager.
JOB_SPECS = {index: (method, TARGET_DTYPE, TARGET_GPU) for index, method in enumerate(('pitome', 'dtome', 'illava', 'ipcv'))}
METHOD_LABELS = {'pitome': 'PiToMe OneVision visual port', 'dtome': 'author DToMe visual merge rules on HF host; no LLM VTU', 'illava': 'author-source iLLaVA OneVision visual branch; no LLM merge', 'ipcv': 'author-scope IPCV global visual top-k/NGR on OneVision; no language pruning'}
METHOD_POINTS = {'pitome': {'ratio': 0.96}, 'dtome': {'checkpoint': 'dtome_324out', 'thresholds': [1.0, 0.94140625, 0.99609375, 0.97265625, 0.94140625, 0.94140625, 0.94140625, 0.94140625, 0.94140625, 0.94140625, 0.94140625, 0.94140625, 0.94140625, 0.94140625, 0.94140625, 0.94140625, 0.94140625, 0.94140625, 0.94140625, 0.94140625, 0.94140625, 0.94140625, 0.94140625, 0.94140625, 0.94140625, 0.94140625], 'provenance': {'repo': 'mikewang/DyMU', 'revision': '1ba4427ac90ca28eff5904fbc419c2381b93a067', 'filename': 'siglip-so400m-patch14-384-tome-324out.pth', 'sha256': '01412a9a98129794f1f9cef814e3c07ecc266c969907de1ee168441a2221e5d6'}}, 'illava': {'zero_based_layers': [5, 6, 7, 8], 'removed_per_layer': 92}, 'ipcv': {'zero_based_prune_layer': 3, 'keep_ratio': 0.85, 'as_layers': 7, 'top_k': 10}}
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
    "tf32": False,
    "journal": "byte_exact_smolvlm_checkpoint_journal",
}
assert sum(DATASET_SIZES.values()) == 32338
assert sum(PROTOCOL['timing_requests_by_task'].values()) == 280
