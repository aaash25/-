# Publication copy: comments/line endings only; algorithm AST unchanged.
# Project migration copy; retain upstream method and host notices. No experiment executed.
# Component licensing and exact source hashes: sources/FILE_DISPOSITIONS.json.
"""LLaVA-1.5's VLMEvalKit conversation and 4.57 image-token alignment."""
from __future__ import annotations

SYSTEM_PROMPT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions. "
)
NATIVE_IMAGE_TOKENS = 576
PITOME_IMAGE_TOKENS = 232  # H299 exact-source Kaggle FP16 active point.


def legacy_llava_prompt(messages: list[dict]) -> str:
    """Match VLMEvalKit LLaVA.chat_inner's user turn, with an assistant colon terminator."""
    content = ""
    image_count = 0
    for item in messages:
        if item["type"] == "text":
            content += item["value"]
        elif item["type"] == "image":
            content += " <image> "
            image_count += 1
        else:
            raise ValueError(f"unsupported LLaVA content type: {item['type']}")
    if image_count < 1 or not content.strip():
        raise ValueError("LLaVA request needs image and text")
    return SYSTEM_PROMPT + "USER: " + content + " ASSISTANT:"


def rewrite_image_runs(ids: list[int], image_id: int,
                       selected_counts: list[int], native_count: int = NATIVE_IMAGE_TOKENS) -> list[int]:
    """Rewrite each image span, retaining all intervening text and image order.

    Adjacent image spans may coalesce into one token run in a tokenizer; the
    frozen native span length makes that split unambiguous.
    """
    if not selected_counts or any(count < 1 for count in selected_counts):
        raise ValueError("one positive selected feature count per image is required")
    output, cursor, image_index = [], 0, 0
    while cursor < len(ids):
        if ids[cursor] != image_id:
            output.append(ids[cursor])
            cursor += 1
            continue
        end = cursor
        while end < len(ids) and ids[end] == image_id:
            end += 1
        span = end - cursor
        if span % native_count:
            raise ValueError(f"image token run is not a multiple of {native_count}: {span}")
        images = span // native_count
        if image_index + images > len(selected_counts):
            raise ValueError("more native image spans than supplied images")
        for count in selected_counts[image_index:image_index + images]:
            output.extend([image_id] * count)
        image_index += images
        cursor = end
    if image_index != len(selected_counts):
        raise ValueError(f"expected {len(selected_counts)} image spans, found {image_index}")
    return output
