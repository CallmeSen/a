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
)
from .data import build_transform


def _class_probabilities(logits: torch.Tensor):
    probs = torch.softmax(logits.float(), dim=-1)
    return probs


def _format_result(logits: torch.Tensor) -> dict:
    probs = _class_probabilities(logits)
    pred_classes = logits.argmax(dim=-1)

    probs = probs.cpu()
    pred_classes = pred_classes.cpu()

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
    model,
    tokenizer,
    return_logits: bool = False,
) -> dict:
    model.eval()
    transform = build_transform(IMAGE_SIZE)

    image = Image.open(image_path).convert("RGB")
    pixel_values = transform(image).unsqueeze(0).to(device, dtype=COMPUTE_DTYPE)

    text_inputs = tokenizer(
        [comment],
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

    logits = outputs["logits"].squeeze(0)
    result = _format_result(logits)

    if return_logits:
        result["logits"] = logits.cpu()

    return result