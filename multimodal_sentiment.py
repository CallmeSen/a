#!/usr/bin/env python3
"""Standalone training script for Multimodal Sentiment Analysis.

Compatible with both python3 and torchrun:
    python3 multimodal_sentiment.py
    torchrun multimodal_sentiment.py
"""

import os
import math

# torchrun sets these env vars automatically; plain python3 leaves them unset
_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
_IS_MAIN_RANK = (_LOCAL_RANK == 0)

# NOTE: CUBLAS_WORKSPACE_CONFIG causes NaN in InternLM2 rotary embedding
# on NVIDIA L40S GPU when using expand() buffers. Must unset BEFORE import torch.
os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

# ============================================================
# Deterministic seeding for reproducibility
# ============================================================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
# NOTE: torch.use_deterministic_algorithms removed because InternLM2 uses
# SDPA/Flash Attention which is non-deterministic by design.
# Forcing deterministic mode causes NaN with large hidden_size (2048).
# ============================================================

from safetensors.torch import save_file, load_file
from sklearn.metrics import precision_score, recall_score, f1_score

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
    build_train_transform,
    compute_class_weights,
    LazyLambdaScheduler,
    train_epoch,
    validate,
    setup_optimizer,
    predict_aspect_sentiment,
    build_tokenizer_and_llm,
)
from src.multitask_model import MultitaskSentimentModel
from src.lora_layers import apply_lora_to_llm, print_lora_summary
from src.data import build_weighted_sampler
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
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    WARMUP_PROPORTION,
    GRADIENT_ACCUMULATION_STEPS,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    ASPECT_LABELS,
    CLASS_LABELS,
    ASPECT2ID,
    NUM_CLASSES,
    USE_LORA,
    USE_MULTITASK,
    USE_WEIGHTED_SAMPLER,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_LR,
)


def main():
    setup_runtime()

    print(f"Device: {device}")
    if os.path.exists(BEST_MODEL_PATH):
        print(f"[INFO] Found existing checkpoint: {BEST_MODEL_PATH}")
        print("      Will resume from checkpoint after training.")

    print(f"Compute dtype: {COMPUTE_DTYPE}")
    print(f"Vision model: {VISION_MODEL_NAME}")
    print(f"LLM model:    {LLM_MODEL_NAME}")
    print(f"Data dir:     {DATA_DIR}")
    print(f"Output dir:   {OUTPUT_DIR}")

    # 1) Load dataset splits
    dataset_splits = load_all_splits(DATA_DIR)

    # Compute class weights from training set for imbalanced CrossEntropyLoss
    class_weights = compute_class_weights(dataset_splits, NUM_CLASSES).to(device)

    # 2) Build tokenizer + LLM base (Qwen2.5-7B-Instruct)
    print(f"\nLoading LLM: {LLM_MODEL_NAME}")
    tokenizer, llm_for_clm, llm_base, num_layers, llm_hidden_size = build_tokenizer_and_llm()
    print(
        f"Tokenizer: padding_side={tokenizer.padding_side}, "
        f"pad_token={tokenizer.pad_token} (id={tokenizer.pad_token_id})"
    )

    # 2b) Apply LoRA to Qwen ForCausalLM if USE_LORA=1
    # NOTE: PEFT wraps llm_for_clm but keeps llm_for_clm.model = original Qwen2Model.
    # QwenLMWrapper traverses qwen_for_casual_lm.model.model... to find Qwen2Model
    # and uses its embed_tokens. LoRA params are injected into Qwen2Model layers.
    if USE_LORA:
        print(f"\n[LoRA] Applying LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
        llm_for_clm = apply_lora_to_llm(
            llm_for_clm,
            r=LORA_R,
            alpha=LORA_ALPHA,
            dropout=LORA_DROPOUT,
        )
        print_lora_summary(llm_for_clm)

    # 3) Build vision + projector
    vision_encoder = VisionEncoder(
        model_name=VISION_MODEL_NAME,
        run_device=device,
        torch_dtype=COMPUTE_DTYPE,
    )
    projector = MLPProjector(
        vision_dim=VISION_HIDDEN_SIZE,
        llm_dim=llm_hidden_size,
    ).to(device)

    # 4) PerceiverResampler
    perceiver_resampler = PerceiverResampler(
        vision_dim=llm_hidden_size,
        num_queries=64,
        num_heads=8,
        expansion=4,
    ).to(device)

    # 5) QwenLMWrapper (visual cross-attention injection)
    use_adapter_layers = list(range(num_layers - 4, num_layers))  # last 4 adapters
    llm_wrapper = QwenLMWrapper(
        qwen_for_casual_lm=llm_for_clm,
        num_layers=num_layers,
        hidden_size=llm_hidden_size,
        num_visual_tokens=64 * MAX_IMAGES,
        use_adapter_layers=use_adapter_layers,
    ).to(device)

    # 6) Full multimodal model (single-task or R8 multitask)
    sentiment_model = MultimodalSentimentModel(
        vision_encoder=vision_encoder,
        projector=projector,
        perceiver_resampler=perceiver_resampler,
        llm_wrapper=llm_wrapper,
        tokenizer=tokenizer,
        num_aspects=len(ASPECT_LABELS),
        num_classes=len(CLASS_LABELS),
    ).to(device)

    # R8: Wrap with MultitaskSentimentModel if USE_MULTITASK=1
    # Note: Soft gate and aspect detection auxiliary head were removed
    # (caused gradient conflicts and no useful signal).
    # MultitaskSentimentModel now wraps the base model without modifications.
    if USE_MULTITASK:
        print("\n[R8] Wrapping model with MultitaskSentimentModel")
        sentiment_model = MultitaskSentimentModel(sentiment_model).to(device)

    total_params, trainable_params_total = sentiment_model.get_trainable_params()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params_total:,}")

    # 7) Build dataloaders
    train_loader_local = _build_train_loader(dataset_splits, tokenizer)
    val_loader_local = _build_val_loader(dataset_splits, tokenizer)
    test_loader_local = _build_test_loader(dataset_splits, tokenizer)

    # 8) Optimizer + scheduler — use higher LR for LoRA params (r5 = 5x base LR)
    optimizer, trainable_params = setup_optimizer(
        sentiment_model,
        LEARNING_RATE,
        WEIGHT_DECAY,
        lora_lr=LORA_LR,
    )

    steps_per_epoch = math.ceil(len(train_loader_local) / GRADIENT_ACCUMULATION_STEPS)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_PROPORTION)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LazyLambdaScheduler(optimizer, lr_lambda)

    print(f"\nTraining config:")
    print(f"  LR: {LEARNING_RATE}")
    print(
        f"  Epochs: {NUM_EPOCHS}, Steps/Epoch: {steps_per_epoch}, "
        f"Total: {total_steps}, Warmup: {warmup_steps}"
    )
    print(f"  Grad accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    trainable_total = sum(p.numel() for p in trainable_params)
    print(f"  Trainable params: {trainable_total:,} params across {len(trainable_params)} tensors")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 9) Sanity forward pass
    sentiment_model.eval()
    batch0 = next(iter(train_loader_local))
    with torch.no_grad():
        pixel_values = batch0["pixel_values"].to(device, dtype=COMPUTE_DTYPE)
        input_ids = batch0["input_ids"].to(device)

        attention_mask = batch0.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        image_counts = batch0.get("image_counts")
        if image_counts is not None:
            image_counts = image_counts.to(device)

        out0 = sentiment_model(
            pixel_values,
            input_ids,
            attention_mask=attention_mask,
            image_counts=image_counts,
        )
        print(f"\n[OK] Sanity forward: logits finite={torch.isfinite(out0['logits']).all().item()}")
        print(f"     logits shape={out0['logits'].shape}")
        print(f"     bad_batch={out0.get('bad_batch', False)}")
    sentiment_model.train()

    # 10) Training loop
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        try:
            train_loss, train_cls = train_epoch(
                sentiment_model,
                train_loader_local,
                optimizer,
                scheduler,
                device,
                trainable_params,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                tokenizer=tokenizer,
                class_weights=class_weights,
                use_multitask=USE_MULTITASK,
            )
            train_losses.append(train_loss)
        except RuntimeError as e:
            if "NaN" in str(e) or "FATAL" in str(e):
                print(f"\n{'='*60}")
                print(f"[FATAL] Training stopped due to NaN loss.")
                print(f"Error: {e}")
                print(f"{'='*60}")
                raise SystemExit(1) from e
            raise

        val_loss, val_cls, macro_f1, val_preds, _ = validate(
            sentiment_model,
            val_loader_local,
            device,
            class_weights=class_weights,
            use_multitask=USE_MULTITASK,
        )
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f} (CLS={train_cls:.4f})")
        print(f"Val Loss:   {val_loss:.4f} (CLS={val_cls:.4f})")
        print(f"Val Macro-F1: {macro_f1:.4f}")

        if not math.isfinite(val_loss) or val_preds is None:
            epochs_without_improvement += 1
        elif macro_f1 > best_macro_f1 + EARLY_STOPPING_MIN_DELTA:
            best_macro_f1 = macro_f1
            epochs_without_improvement = 0

            trainable_param_names = {
                name for name, p in sentiment_model.named_parameters() if p.requires_grad
            }
            trainable_state = {
                name: tensor.detach().cpu()
                for name, tensor in sentiment_model.state_dict().items()
                if name in trainable_param_names
            }
            save_file(trainable_state, BEST_MODEL_PATH)
            torch.save(
                {
                    "epoch": epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "macro_f1": macro_f1,
                    "num_aspects": len(ASPECT_LABELS),
                    "num_classes": len(CLASS_LABELS),
                    "aspect_labels": ASPECT_LABELS,
                    "class_labels": CLASS_LABELS,
                    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                    "trainable_param_names": sorted(trainable_param_names),
                },
                TRAINING_INFO_PATH,
                _use_new_zipfile_serialization=True,
            )
            print(f"[SAVED] Best model macro-F1={macro_f1:.4f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("[EARLY STOP] triggered")
            break

        # 11) Load best and evaluate on test set
    if os.path.exists(BEST_MODEL_PATH):
        trainable_state = load_file(BEST_MODEL_PATH)
        trainable_state = {k: v.to(device) for k, v in trainable_state.items()}
        sentiment_model.load_state_dict(trainable_state, strict=False)

    test_loss, test_cls, test_macro_f1, test_preds, test_labels = validate(
        sentiment_model,
        test_loader_local,
        device,
        class_weights=class_weights,
        use_multitask=USE_MULTITASK,
    )
    print(f"\n=== Test Set ===")
    print(f"Test Loss: {test_loss:.4f} (CLS={test_cls:.4f})")
    print(f"Test Macro-F1: {test_macro_f1:.4f}")

    if test_preds is not None:
        y_true = test_labels.numpy().ravel()
        y_pred = test_preds.numpy().ravel()
        macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall:   {macro_recall:.4f}")
        print(f"Macro F1:       {macro_f1:.4f}")

    # 12) Demo inference on first test sample
    demo_test = SentimentDataset(
        dataset_splits["test"],
        IMAGE_DIR,
        ASPECT2ID,
        transform=build_transform(IMAGE_SIZE),
    )
    if len(demo_test.samples) > 0:
        demo_sample = demo_test.samples[0]
        demo_image_path = demo_sample["image_paths"][0]
        demo_comment = demo_sample["comment"]
        demo_true_label = demo_sample["sentiment"]

        print(f"\n=== Demo Inference ===")
        print(f"Comment: {demo_comment}")
        print(f"True label: {demo_true_label}")
        print(f"\n--- Aspect-level predictions ---")

        for demo_aspect_name in ASPECT_LABELS:
            result = predict_aspect_sentiment(
                demo_image_path,
                demo_comment,
                demo_aspect_name,
                sentiment_model,
                tokenizer,
            )
            print(f"\nAspect: {demo_aspect_name}")
            print(f"  Predicted: {result['predicted_label']}")
            print(f"  Probabilities: {result['probabilities']}")


def _build_train_loader(dataset_splits, tokenizer):
    train_dataset = SentimentDataset(
        dataset_splits["train"],
        IMAGE_DIR,
        ASPECT2ID,
        transform=build_train_transform(IMAGE_SIZE),
    )
    collate_fn = make_collate_fn(tokenizer)
    g = torch.Generator()
    g.manual_seed(SEED)

    if USE_WEIGHTED_SAMPLER:
        # R7: WeightedRandomSampler — oversamples minority sentiment samples
        sampler = build_weighted_sampler(train_dataset, minority_upsample_ratio=2.0)
        return DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
        )
    else:
        return DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            generator=g,
        )


def _build_val_loader(dataset_splits, tokenizer):
    dev_dataset = SentimentDataset(
        dataset_splits["dev"],
        IMAGE_DIR,
        ASPECT2ID,
        transform=build_transform(IMAGE_SIZE),
    )
    collate_fn = make_collate_fn(tokenizer)
    return DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
    )


def _build_test_loader(dataset_splits, tokenizer):
    test_dataset_local = SentimentDataset(
        dataset_splits["test"],
        IMAGE_DIR,
        ASPECT2ID,
        transform=build_transform(IMAGE_SIZE),
    )
    collate_fn = make_collate_fn(tokenizer)
    return DataLoader(
        test_dataset_local,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
    )


def test_dataset():
    return SentimentDataset(
        load_all_splits(DATA_DIR)["test"],
        IMAGE_DIR,
        ASPECT2ID,
        transform=build_transform(IMAGE_SIZE),
    )


if __name__ == "__main__":
    main()