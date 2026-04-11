#!/usr/bin/env python3
"""Standalone inference script for Multimodal Sentiment Analysis.

Chạy: python inference.py [--split train|dev|test] [--num-samples N]

Load checkpoint đã train và đánh giá trên tập train/dev/test.
"""
import argparse
import os
import json

os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

from src import (
    VisionEncoder,
    MLPProjector,
    MultimodalSentimentModel,
    PerceiverResampler,
    QwenLMWrapper,
    SentimentDataset,
    make_collate_fn,
    load_all_splits,
    build_transform,
    build_tokenizer_and_llm,
    validate,
)
from src.config import (
    setup_runtime,
    device,
    COMPUTE_DTYPE,
    VISION_MODEL_NAME,
    VISION_HIDDEN_SIZE,
    LLM_MODEL_NAME,
    IMAGE_SIZE,
    MAX_IMAGES,
    DATA_DIR,
    IMAGE_DIR,
    OUTPUT_DIR,
    BEST_MODEL_PATH,
    TRAINING_INFO_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    ASPECT_LABELS,
    CLASS_LABELS,
    ASPECT2ID,
    NUM_CLASSES,
    USE_LORA,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
)
from src.lora_layers import apply_lora_to_llm


def build_model_from_checkpoint(tokenizer):
    """Build full model architecture matching the saved checkpoint."""
    _, llm_for_clm, llm_base, num_layers, llm_hidden_size = build_tokenizer_and_llm()

    if USE_LORA:
        llm_for_clm = apply_lora_to_llm(
            llm_for_clm,
            r=LORA_R,
            alpha=LORA_ALPHA,
            dropout=LORA_DROPOUT,
        )

    vision_encoder = VisionEncoder(
        model_name=VISION_MODEL_NAME,
        run_device=device,
        torch_dtype=COMPUTE_DTYPE,
    )
    projector = MLPProjector(
        vision_dim=VISION_HIDDEN_SIZE,
        llm_dim=llm_hidden_size,
    ).to(device)
    perceiver_resampler = PerceiverResampler(
        vision_dim=llm_hidden_size,
        num_queries=64,
        num_heads=8,
        expansion=4,
    ).to(device)
    use_adapter_layers = list(range(num_layers // 2, num_layers))
    llm_wrapper = QwenLMWrapper(
        qwen_for_casual_lm=llm_for_clm,
        num_layers=num_layers,
        hidden_size=llm_hidden_size,
        num_visual_tokens=64 * MAX_IMAGES,
        use_adapter_layers=use_adapter_layers,
    ).to(device)

    model = MultimodalSentimentModel(
        vision_encoder=vision_encoder,
        projector=projector,
        perceiver_resampler=perceiver_resampler,
        llm_wrapper=llm_wrapper,
        tokenizer=tokenizer,
        num_aspects=len(ASPECT_LABELS),
        num_classes=len(CLASS_LABELS),
    ).to(device)

    return model


def load_checkpoint(model, checkpoint_path: str):
    """Load trainable params from checkpoint into model."""
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    state = load_file(checkpoint_path)
    state = {k: v.to(device) for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded {len(state)} state dict entries")
    if missing:
        print(f"  Missing keys (expected - non-trainable): {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    return model


def run_inference(model, dataloader, device, class_weights=None, use_multitask=False):
    """Run inference and return predictions + labels."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device, dtype=COMPUTE_DTYPE)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            image_counts = batch.get("image_counts")
            if image_counts is not None:
                image_counts = image_counts.to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                pixel_values,
                input_ids,
                attention_mask=attention_mask,
                image_counts=image_counts,
            )

            logits = outputs["logits"]
            # squeeze to [B, num_aspects, num_classes]
            if logits.dim() == 4:
                logits = logits.squeeze(1)
            if logits.dim() == 3:
                logits = logits.squeeze(1)

            # Flatten
            logits_flat = logits.view(-1, NUM_CLASSES)
            labels_flat = labels.view(-1)

            # Loss
            if class_weights is not None:
                loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
                loss = loss_fn(logits_flat, labels_flat)
                total_loss += loss.item()

            # Predictions
            preds = logits_flat.argmax(dim=-1)
            probs = torch.softmax(logits_flat.float(), dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels_flat.cpu())
            all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probs = torch.cat(all_probs, dim=0)

    avg_loss = total_loss / max(len(dataloader), 1)
    return all_preds, all_labels, all_probs, avg_loss


def print_metrics(y_true, y_pred, loss, split_name, class_labels):
    """Print classification metrics."""
    print(f"\n{'='*60}")
    print(f"=== {split_name.upper()} RESULTS ===")
    print(f"{'='*60}")
    print(f"Loss: {loss:.4f}")
    print(f"Macro-F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Macro Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Macro Recall: {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"\nPer-class metrics:")
    print(classification_report(y_true, y_pred, target_names=class_labels, digits=4, zero_division=0))

    # Confusion matrix summary
    from collections import Counter
    true_counts = Counter(y_true.tolist())
    pred_counts = Counter(y_pred.tolist())
    print("Label distribution (true):")
    for i, label in enumerate(class_labels):
        print(f"  {label}: {true_counts.get(i, 0)}")
    print("Prediction distribution:")
    for i, label in enumerate(class_labels):
        print(f"  {label}: {pred_counts.get(i, 0)}")


def main():
    parser = argparse.ArgumentParser(description="Standalone Inference for Multimodal Sentiment")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit number of samples (for quick test)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference")
    args = parser.parse_args()

    setup_runtime()
    print(f"Device: {device}")
    print(f"Compute dtype: {COMPUTE_DTYPE}")

    # Check checkpoint
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"[ERROR] No checkpoint found: {BEST_MODEL_PATH}")
        print("  Run training first: python multimodal_sentiment.py")
        return

    if not os.path.exists(TRAINING_INFO_PATH):
        print(f"[ERROR] No training info found: {TRAINING_INFO_PATH}")
        return

    # Load training info
    info = torch.load(TRAINING_INFO_PATH, map_location="cpu", weights_only=False)
    print(f"[INFO] Checkpoint from epoch {info['epoch']+1}, val macro-f1={info['macro_f1']:.4f}")
    print(f"[INFO] Classes: {info['class_labels']}")
    print(f"[INFO] Aspects: {info['aspect_labels']}")

    # Build model
    tokenizer, _, _, _, _ = build_tokenizer_and_llm()
    print("\n[INFO] Building model architecture...")
    model = build_model_from_checkpoint(tokenizer)

    # Load checkpoint
    model = load_checkpoint(model, BEST_MODEL_PATH)
    model.eval()

    # Load dataset
    dataset_splits = load_all_splits(DATA_DIR)
    split_data = dataset_splits.get(args.split, [])
    if not split_data:
        print(f"[ERROR] No {args.split} data found")
        return

    if args.num_samples:
        split_data = split_data[: args.num_samples]

    print(f"\n[INFO] Loading {args.split} dataset: {len(split_data)} samples")
    dataset = SentimentDataset(
        split_data,
        IMAGE_DIR,
        ASPECT2ID,
        transform=build_transform(IMAGE_SIZE),
    )
    collate_fn = make_collate_fn(tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
    )

    # Compute class weights
    from src.data import compute_class_weights
    class_weights = compute_class_weights(dataset_splits, NUM_CLASSES).to(device)

    # Run inference
    print(f"\n[INFO] Running inference on {args.split}...")
    preds, labels, probs, loss = run_inference(model, dataloader, device, class_weights)

    # Print metrics
    y_true = labels.numpy().ravel()
    y_pred = preds.numpy().ravel()
    print_metrics(y_true, y_pred, loss, args.split, CLASS_LABELS)

    # Save predictions to file
    results_path = os.path.join(OUTPUT_DIR, f"inference_{args.split}_results.json")
    results = {
        "split": args.split,
        "num_samples": len(y_true),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "predictions": y_pred.tolist(),
        "labels": y_true.tolist(),
        "probabilities": probs.numpy().tolist(),
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Results saved to: {results_path}")


if __name__ == "__main__":
    main()
