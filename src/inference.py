"""Inference utilities for Multimodal Sentiment Analysis."""

from typing import Optional

from PIL import Image
import torch

from .config import (
    device,
    COMPUTE_DTYPE,
    IMAGE_SIZE,
    NUM_CLASSES,
    NUM_ASPECTS,
    ID2ASPECT,
    ID2CLASS,
    MAX_TEXT_LENGTH,
    ASPECT_START,
    ASPECT_END,
)
from .data import build_transform


def _build_aspect_text(comment: str, aspect_name: str) -> str:
    """Wrap comment with aspect instruction tokens per diagram."""
    return f"{ASPECT_START}{aspect_name}{ASPECT_END} {comment}"


def _class_probabilities(logits: torch.Tensor):
    probs = torch.softmax(logits.float(), dim=-1)
    return probs


def _format_result(logits: torch.Tensor, single_aspect: bool = False) -> dict:
    """Format model output into human-readable results.

    Args:
        logits: [1, 4] tensor from single-aspect model (after squeeze)
        single_aspect: if True, logits is [1, 4] (new diagram architecture)
    """
    probs = torch.softmax(logits.float(), dim=-1)
    pred_classes = logits.argmax(dim=-1)

    probs = probs.cpu()
    pred_classes = pred_classes.cpu()

    if single_aspect:
        pred_class = pred_classes[0, 0].item()
        pred_label = ID2CLASS[pred_class]
        return {
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "probabilities": {ID2CLASS[c]: probs[0, 0, c].item() for c in range(NUM_CLASSES)},
        }

    # Multi-aspect legacy path (kept for reference)
    detected_aspects = []
    aspect_sentiments = {}
    detailed = {}

    for a_idx in range(NUM_ASPECTS):
        aspect_name = ID2ASPECT[a_idx]
        pred_class = pred_classes[a_idx].item()
        pred_label = ID2CLASS[pred_class]

        detailed[aspect_name] = {
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "probabilities": {ID2CLASS[c]: probs[a_idx, c].item() for c in range(NUM_CLASSES)},
        }

        if pred_class > 0:
            detected_aspects.append(aspect_name)
            aspect_sentiments[aspect_name] = pred_label

    return {
        "detected_aspects": detected_aspects,
        "aspect_sentiments": aspect_sentiments,
        "detailed": detailed,
    }


def predict_aspect_sentiment(
    image_path: str,
    comment: str,
    aspect_name: str,
    model,
    tokenizer,
    return_logits: bool = False,
) -> dict:
    """Predict sentiment for a single aspect.

    Diagram-compliant: builds <ASP>aspect</ASP> prompt template,
    extracts h_a via mean pooling over span, dot-product attention pooling,
    then classifies.
    """
    model.eval()
    transform = build_transform(IMAGE_SIZE)

    # Build aspect-prompted text
    aspect_text = _build_aspect_text(comment, aspect_name)

    image = Image.open(image_path).convert("RGB")
    pixel_values = transform(image).unsqueeze(0).to(device, dtype=COMPUTE_DTYPE)

    text_inputs = tokenizer(
        [aspect_text],
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_LENGTH,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    image_counts = torch.tensor([1], device=device, dtype=torch.long)

    with torch.no_grad():
        outputs = model(
            pixel_values,
            input_ids,
            attention_mask=attention_mask,
            image_counts=image_counts,
        )

    if outputs.get("bad_batch", False):
        raise RuntimeError("Model returned bad_batch during inference")

    # logits shape: [1, 1, 4] → squeeze to [1, 4]
    logits = outputs["logits"].squeeze(0)  # [1, 4]
    result = _format_result(logits, single_aspect=True)

    if return_logits:
        result["logits"] = logits.cpu()

    return result
