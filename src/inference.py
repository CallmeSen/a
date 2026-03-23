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


def _combine_acd_asc_predictions(acd_logits: torch.Tensor, asc_logits: torch.Tensor) -> torch.Tensor:
    pred_presence = (acd_logits > 0).long()
    pred_sentiment = asc_logits.argmax(dim=-1) + 1
    return pred_presence * pred_sentiment


def _acd_asc_class_probabilities(acd_logits: torch.Tensor, asc_logits: torch.Tensor):
    presence_probs = torch.sigmoid(acd_logits.float())
    sentiment_probs = torch.softmax(asc_logits.float(), dim=-1)

    combined_probs = torch.zeros(*acd_logits.shape, NUM_CLASSES, device=acd_logits.device, dtype=torch.float32)
    combined_probs[..., 0] = 1.0 - presence_probs
    combined_probs[..., 1:] = presence_probs.unsqueeze(-1) * sentiment_probs
    return combined_probs, presence_probs, sentiment_probs


def _format_acd_asc_result(acd_logits: torch.Tensor, asc_logits: torch.Tensor) -> dict:
    combined_probs, presence_probs, sentiment_probs = _acd_asc_class_probabilities(acd_logits, asc_logits)
    pred_classes = _combine_acd_asc_predictions(acd_logits, asc_logits)

    combined_probs = combined_probs.cpu()
    presence_probs = presence_probs.cpu()
    sentiment_probs = sentiment_probs.cpu()
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
            "presence_probability": presence_probs[a_idx].item(),
            "probabilities": {ID2CLASS[c]: combined_probs[a_idx, c].item() for c in range(NUM_CLASSES)},
            "sentiment_probabilities": {
                "Negative": sentiment_probs[a_idx, 0].item(),
                "Neutral": sentiment_probs[a_idx, 1].item(),
                "Positive": sentiment_probs[a_idx, 2].item(),
            },
        }

        if pred_class > 0:
            detected_aspects.append(aspect_name)
            aspect_sentiments[aspect_name] = pred_label

    return {"detected_aspects": detected_aspects, "aspect_sentiments": aspect_sentiments, "detailed": detailed}


def predict_aspect_sentiment(image_path: str, comment: str, model, tokenizer) -> dict:
    model.eval()
    transform = build_transform(IMAGE_SIZE)

    image = Image.open(image_path).convert("RGB")
    pixel_values = transform(image).unsqueeze(0).to(device, dtype=COMPUTE_DTYPE)

    text_inputs = tokenizer([comment], padding=True, truncation=True, max_length=MAX_TEXT_LENGTH, return_tensors="pt")
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    with torch.no_grad():
        outputs = model(pixel_values, input_ids, attention_mask)

    return _format_acd_asc_result(outputs["acd_logits"].squeeze(0), outputs["asc_logits"].squeeze(0))
