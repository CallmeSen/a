import os
import argparse
from pathlib import Path
from typing import List, Any, Optional, Dict

import torch
from PIL import Image

from src.config import (
    LLM_NAME, VIT_NAME, ASPECT_LABELS, NUM_ASPECTS, IMAGE_SIZE, MAX_TEXT_LEN,
    DATA_DIR, OUTPUT_DIR,
)
from src.aspect_model import MultimodalACSAModel
from transformers import AutoTokenizer, AutoProcessor


SENTIMENT_LABELS = ["Irrelative", "Negative", "Neutral", "Positive"]

_siglip_processor = None

def _get_siglip_processor():
    global _siglip_processor
    if _siglip_processor is None:
        _siglip_processor = AutoProcessor.from_pretrained(VIT_NAME)
    return _siglip_processor


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with MultimodalSentimentModel")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to single checkpoint .pt file (or outputs dir for best_checkpoint.pt)")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", dest="use_lora", action="store_false")
    return parser.parse_args()


def load_model(checkpoint_path: Optional[str], device: str = "cuda") -> MultimodalACSAModel:
    """
    Load trained MultimodalSentimentModel model (single model, all 6 aspects).
    If checkpoint_path is None, returns model with random weights (for testing).
    """
    from training import load_model_weights

    model = MultimodalACSAModel(use_lora=True).to(device)
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if ckpt.is_dir():
            ckpt = ckpt / "best_checkpoint.pt"
        if ckpt.exists():
            load_model_weights(model, str(ckpt), device=device)
            print(f"Loaded weights from {ckpt}")
        else:
            print(f"Warning: checkpoint not found at {ckpt}, using random weights")
    return model


def preprocess_image(
    image_path: str,
    size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """
    Load and preprocess a single image using SigLIP processor.
    Returns [3, H, W] tensor normalized by the processor.
    """
    processor = _get_siglip_processor()
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"][0]  # [3, H, W]
    return pixel_values


def preprocess_images(
    image_paths: List[str],
    max_images: int = 7,
    size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """
    Load and stack multiple images.
    Returns [max_images, 3, H, W] tensor.
    """
    images = []
    for path in image_paths[:max_images]:
        try:
            img_tensor = preprocess_image(path, size=size)
            images.append(img_tensor)
        except Exception:
            pass

    while len(images) < max_images:
        images.append(torch.zeros(3, size, size))

    return torch.stack(images)


def predict_all_aspects(
    model: MultimodalACSAModel,
    tokenizer,
    comment: str,
    image_paths: List[str],
    roi_data: Optional[List[Dict[str, Any]]] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Predict sentiments for all 6 aspects in a single forward pass.
    roi_data: list of roi dicts per image, same format as dataset.
        If None, RoI encoder will return zeros.
    """
    model.eval()

    pixel_values = preprocess_images(image_paths).to(device)

    aspect_labels = {i: 0 for i in range(NUM_ASPECTS)}

    # Default roi_data: empty boxes for each image, wrapped as per-sample list
    if roi_data is None:
        roi_data = [[{"boxes": [], "labels": []} for _ in image_paths[:7]]]

    encodings = tokenizer(
        comment,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_LEN,
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        logits = model.inference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values.unsqueeze(0),
            roi_data=roi_data,
            aspect_labels=[aspect_labels],
        )

    results = {}
    for asp_idx, aspect_name in enumerate(ASPECT_LABELS):
        probs = logits[asp_idx].cpu().tolist()
        pred_id = int(torch.argmax(logits[asp_idx]).item())

        results[aspect_name] = {
            "prediction": SENTIMENT_LABELS[pred_id],
            "prediction_id": pred_id,
            "confidence": probs[pred_id],
            "probabilities": dict(zip(SENTIMENT_LABELS, probs)),
        }

    return results


def predict_batch(
    model: MultimodalACSAModel,
    tokenizer,
    comments: List[str],
    image_paths_list: List[List[str]],
    device: str = "cuda",
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """
    Predict sentiments for a batch of samples (all 6 aspects each).
    """
    model.eval()

    results = []
    for i in range(0, len(comments), batch_size):
        batch_comments = comments[i : i + batch_size]
        batch_images = image_paths_list[i : i + batch_size]
        B = len(batch_comments)

        pixel_values_list = []
        for paths in batch_images:
            pv = preprocess_images(paths)
            pixel_values_list.append(pv)
        pixel_values = torch.stack(pixel_values_list).to(device)

        aspect_labels = [{j: 0 for j in range(NUM_ASPECTS)} for _ in range(B)]

        roi_data = [[{"boxes": [], "labels": []} for _ in range(len(paths))] for paths in batch_images]

        encodings = tokenizer(
            batch_comments,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LEN,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            logits = model.inference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                roi_data=roi_data,
                aspect_labels=aspect_labels,
            )

        logits_per_sample = logits.view(B, NUM_ASPECTS, -1)

        for b, comment in enumerate(batch_comments):
            sample_result = {}
            for asp_idx, aspect_name in enumerate(ASPECT_LABELS):
                probs = logits_per_sample[b, asp_idx].cpu().tolist()
                pred_id = int(torch.argmax(logits_per_sample[b, asp_idx]).item())

                sample_result[aspect_name] = {
                    "prediction": SENTIMENT_LABELS[pred_id],
                    "prediction_id": pred_id,
                    "confidence": probs[pred_id],
                    "probabilities": dict(zip(SENTIMENT_LABELS, probs)),
                }
            results.append(sample_result)

    return results


def main():
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("MultimodalACSAModel Inference (encode-once, aspect-loop)")
    print(f"Device: {device}")

    print("Loading model...")
    model = load_model(args.checkpoint, device)

    print("\nEnter a review and image path to get predictions for all 6 aspects.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            comment = input("Review text: ").strip()
            if comment.lower() in ["quit", "exit", "q"]:
                break
            if not comment:
                continue

            image_path = input("Image path (or press Enter for no image): ").strip()
            image_paths = [image_path] if image_path else []

            results = predict_all_aspects(
                model=model,
                tokenizer=tokenizer,
                comment=comment,
                image_paths=image_paths,
                roi_data=None,
                device=str(device),
            )

            print("\nPredictions:")
            for aspect, result in results.items():
                print(f"  {aspect}: {result['prediction']} "
                      f"(confidence: {result['confidence']:.3f})")

        except (KeyboardInterrupt, EOFError):
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
