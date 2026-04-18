import os
import gc
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from src.config import (
    LLM_NAME, VIT_NAME,
    ASPECT_LABELS, NUM_ASPECTS,
    BATCH_SIZE, GRADIENT_ACCUMULATION, MAX_EPOCHS, EARLY_STOPPING_PATIENCE,
    LR_LORA, LR_OTHER, WEIGHT_DECAY, WARMUP_RATIO,
    DATA_DIR, OUTPUT_DIR,
    MAX_TEXT_LEN,
)
from src.aspect_model import MultimodalACSAModel
from src.data import MultimodalSentimentDataset, collate_fn


def ensure_base_models_cached():
    """
    Download và cache base models 1 lần duy nhất.
    Gọi trước khi bắt đầu training để tránh download lại nhiều lần.
    """
    from huggingface_hub import snapshot_download
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("Ensuring base models are cached...")

    print(f"  Caching LLM: {LLM_NAME}")
    snapshot_download(LLM_NAME, local_files_only=False)

    print(f"  Caching SigLIP: {VIT_NAME}")
    snapshot_download(VIT_NAME, local_files_only=False)

    print("All base models cached successfully.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MultimodalSentimentModel multi-aspect model")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gradient_accumulation", type=int, default=GRADIENT_ACCUMULATION)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--early_stopping_patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--lr_lora", type=float, default=LR_LORA)
    parser.add_argument("--lr_other", type=float, default=LR_OTHER)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup_ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", dest="use_lora", action="store_false")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer():
    """Load tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_optimizer(model, lr_lora: float, lr_other: float, weight_decay: float):
    """Build optimizer with two param groups: LoRA and other params."""
    lora_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name or "lora_B" in name:
            lora_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": lr_other, "weight_decay": weight_decay},
        {"params": lora_params, "lr": lr_lora, "weight_decay": 0.0},
    ])
    return optimizer


def build_scheduler(optimizer, num_training_steps: int, warmup_ratio: float):
    """Build linear warmup + cosine decay scheduler."""
    warmup_steps = int(num_training_steps * warmup_ratio)
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )


def train_epoch(
    model: MultimodalACSAModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    tokenizer,
    gradient_accumulation: int,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(pbar):
        comments = batch["comments"]
        pixel_values = batch["pixel_values"].to(device)
        aspect_labels = batch["aspect_labels"]

        encodings = tokenizer(
            comments,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LEN,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        roi_data = batch["roi_data"]

        with autocast(dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                roi_data=roi_data,
                aspect_labels=aspect_labels,
            )
            loss = outputs["loss"] / gradient_accumulation

        scaler.scale(loss).backward()
        total_loss += loss.item() * gradient_accumulation

        if (step + 1) % gradient_accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix({"loss": f"{total_loss / (num_batches + 1):.4f}"})
        num_batches += 1

    if num_batches % gradient_accumulation != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / max(num_batches, 1)


def eval_epoch(
    model: MultimodalACSAModel,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
) -> dict:
    """
    Evaluate model on dev/test set.
    Returns dict with metrics overall and per-aspect.
    """
    model.eval()
    aspect_preds = {i: [] for i in range(NUM_ASPECTS)}
    aspect_labels_acc = {i: [] for i in range(NUM_ASPECTS)}
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            comments = batch["comments"]
            pixel_values = batch["pixel_values"].to(device)
            aspect_labels = batch["aspect_labels"]

            encodings = tokenizer(
                comments,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TEXT_LEN,
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            roi_data = batch["roi_data"]

            with autocast(dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    roi_data=roi_data,
                    aspect_labels=aspect_labels,
                )

            total_loss += outputs["loss"].item()

            logits = outputs["logits"]  # [B*6, 4]
            targets = outputs["targets"]  # [B*6]

            B = input_ids.size(0)
            for asp_idx in range(NUM_ASPECTS):
                start = asp_idx * B
                end = start + B
                preds = logits[start:end].argmax(dim=-1).cpu().tolist()
                labels = targets[start:end].cpu().tolist()
                aspect_preds[asp_idx].extend(preds)
                aspect_labels_acc[asp_idx].extend(labels)

            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    all_preds = []
    all_labels = []
    for asp_idx in range(NUM_ASPECTS):
        all_preds.extend(aspect_preds[asp_idx])
        all_labels.extend(aspect_labels_acc[asp_idx])

    overall_f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    overall_f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    overall_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    overall_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    overall_accuracy = accuracy_score(all_labels, all_preds)

    per_aspect = {}
    for asp_idx, aspect_name in enumerate(ASPECT_LABELS):
        preds = aspect_preds[asp_idx]
        labels = aspect_labels_acc[asp_idx]
        if len(set(labels)) > 0:
            per_aspect[aspect_name] = {
                "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
                "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
                "precision": precision_score(labels, preds, average="macro", zero_division=0),
                "recall": recall_score(labels, preds, average="macro", zero_division=0),
                "accuracy": accuracy_score(labels, preds),
                "f1_per_class": f1_score(labels, preds, average=None, zero_division=0).tolist(),
            }
        else:
            per_aspect[aspect_name] = {
                "f1_macro": 0.0, "f1_weighted": 0.0,
                "precision": 0.0, "recall": 0.0, "accuracy": 0.0,
                "f1_per_class": [0.0, 0.0, 0.0, 0.0],
            }

    return {
        "loss": avg_loss,
        "overall_f1_macro": overall_f1_macro,
        "overall_f1_weighted": overall_f1_weighted,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_accuracy": overall_accuracy,
        "per_aspect": per_aspect,
    }


SENTIMENT_LABELS = ["Irrelative", "Negative", "Neutral", "Positive"]


def test_epoch(
    model: MultimodalACSAModel,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
) -> dict:
    """
    Evaluate on test set and collect per-sample predictions.
    Returns dict with metrics + list of per-sample predictions.
    """
    model.eval()
    aspect_preds = {i: [] for i in range(NUM_ASPECTS)}
    aspect_labels_acc = {i: [] for i in range(NUM_ASPECTS)}
    total_loss = 0.0
    num_batches = 0

    # Collect per-sample data for test_result.json
    all_comments = []
    all_image_names = []
    all_sample_preds = []   # list of dicts: one per sample
    all_sample_labels = []  # ground truth per sample

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Test Evaluation")
        for batch_idx, batch in enumerate(pbar):
            comments = batch["comments"]
            pixel_values = batch["pixel_values"].to(device)
            aspect_labels = batch["aspect_labels"]
            image_names_batch = batch.get("image_names", [[] for _ in comments])

            encodings = tokenizer(
                comments,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TEXT_LEN,
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            roi_data = batch["roi_data"]

            with autocast(dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    roi_data=roi_data,
                    aspect_labels=aspect_labels,
                )

            total_loss += outputs["loss"].item()

            logits = outputs["logits"]  # [B*6, 4]
            targets = outputs["targets"]  # [B*6]

            B = input_ids.size(0)
            for asp_idx in range(NUM_ASPECTS):
                start = asp_idx * B
                end = start + B
                preds = logits[start:end].argmax(dim=-1).cpu().tolist()
                labels = targets[start:end].cpu().tolist()
                aspect_preds[asp_idx].extend(preds)
                aspect_labels_acc[asp_idx].extend(labels)

            # Per-sample predictions
            logits_per_sample = logits.view(B, NUM_ASPECTS, -1)  # [B, 6, 4]
            probs_per_sample = torch.softmax(logits_per_sample, dim=-1).cpu().tolist()

            for b in range(B):
                sample_preds = {}
                sample_labels = {}
                for asp_idx, aspect_name in enumerate(ASPECT_LABELS):
                    pred_id = int(logits_per_sample[b, asp_idx].argmax().item())
                    prob = probs_per_sample[b, asp_idx]
                    true_label = int(targets[asp_idx * B + b].item())
                    sample_preds[aspect_name] = {
                        "prediction": SENTIMENT_LABELS[pred_id],
                        "prediction_id": pred_id,
                        "confidence": float(prob[pred_id]),
                        "probabilities": {
                            SENTIMENT_LABELS[c]: float(prob[c]) for c in range(4)
                        },
                    }
                    sample_labels[aspect_name] = SENTIMENT_LABELS[true_label]

                all_sample_preds.append({
                    "comment": comments[b],
                    "image_names": image_names_batch[b],
                    "predictions": sample_preds,
                })
                all_sample_labels.append({"aspect_labels": sample_labels})

            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    all_preds = []
    all_labels = []
    for asp_idx in range(NUM_ASPECTS):
        all_preds.extend(aspect_preds[asp_idx])
        all_labels.extend(aspect_labels_acc[asp_idx])

    overall_f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    overall_f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    overall_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    overall_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    overall_accuracy = accuracy_score(all_labels, all_preds)

    per_aspect = {}
    for asp_idx, aspect_name in enumerate(ASPECT_LABELS):
        preds = aspect_preds[asp_idx]
        labels = aspect_labels_acc[asp_idx]
        if len(set(labels)) > 0:
            per_aspect[aspect_name] = {
                "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
                "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
                "precision": precision_score(labels, preds, average="macro", zero_division=0),
                "recall": recall_score(labels, preds, average="macro", zero_division=0),
                "accuracy": accuracy_score(labels, preds),
                "f1_per_class": f1_score(labels, preds, average=None, zero_division=0).tolist(),
            }
        else:
            per_aspect[aspect_name] = {
                "f1_macro": 0.0, "f1_weighted": 0.0,
                "precision": 0.0, "recall": 0.0, "accuracy": 0.0,
                "f1_per_class": [0.0, 0.0, 0.0, 0.0],
            }

    return {
        "loss": avg_loss,
        "overall_f1_macro": overall_f1_macro,
        "overall_f1_weighted": overall_f1_weighted,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_accuracy": overall_accuracy,
        "per_aspect": per_aspect,
        "per_sample_predictions": all_sample_preds,
    }


def save_checkpoint(
    model: MultimodalACSAModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_f1: float,
    path: str,
):
    """
    Save full training state + full model state dict (single model).
    """
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    state = {
        "epoch": epoch,
        "best_f1": best_f1,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(state, path)

    def _flatten_with_prefix(module_state, prefix):
        return {f"{prefix}.{k}": v for k, v in module_state.items()}

    full_state = {}
    full_state.update(_flatten_with_prefix(model.mlp_projector.state_dict(), "mlp_projector"))
    full_state.update(_flatten_with_prefix(model.roi_projector.state_dict(), "roi_projector"))
    full_state.update(_flatten_with_prefix(model.head.state_dict(), "head"))
    full_state.update(_flatten_with_prefix(model.aspect_queries.state_dict(), "aspect_queries"))
    full_state.update(_flatten_with_prefix(model.text_retriever.state_dict(), "text_retriever"))
    full_state.update(_flatten_with_prefix(model.img_retriever.state_dict(), "img_retriever"))
    full_state.update(_flatten_with_prefix(model.roi_retriever.state_dict(), "roi_retriever"))
    full_state.update(_flatten_with_prefix(model.gated_fusion.state_dict(), "gated_fusion"))
    full_state.update(_flatten_with_prefix(model.img_sum_projector.state_dict(), "img_sum_projector"))
    for k, v in get_peft_model_state_dict(model.llm).items():
        full_state[f"lora.{k}"] = v
    safe_path = path.replace(".pt", "_model.safetensors")
    save_file(full_state, safe_path)


def load_model_weights(model: MultimodalACSAModel, ckpt_path: str, device: str = "cuda"):
    """
    Load full model weights từ safetensors — single model checkpoint.
    Tries best_model.safetensors first, then falls back to ckpt_path.
    """
    from safetensors import safe_open
    from peft import set_peft_model_state_dict

    # Load best_checkpoint_model.safetensors
    best_safetensor_path = ckpt_path.replace(".pt", "_model.safetensors")
    if not os.path.exists(best_safetensor_path):
        raise FileNotFoundError(f"Model safetensors not found: {best_safetensor_path}")

    loaded_state = {}
    with safe_open(safetensor_path, framework="pt", device=device) as f:
        for key in f.keys():
            loaded_state[key] = f.get_tensor(key)

    def _extract_sub(state, prefix):
        return {k.replace(prefix + ".", ""): v for k, v in state.items() if k.startswith(prefix + ".")}

    lora_state = {k.replace("lora.", ""): v for k, v in loaded_state.items() if k.startswith("lora.")}
    if lora_state:
        set_peft_model_state_dict(model.llm, lora_state)

    mlp = _extract_sub(loaded_state, "mlp_projector")
    if mlp:
        model.mlp_projector.load_state_dict(mlp, strict=False)
    roi = _extract_sub(loaded_state, "roi_projector")
    if roi:
        model.roi_projector.load_state_dict(roi, strict=False)
    head = _extract_sub(loaded_state, "head")
    if head:
        model.head.load_state_dict(head, strict=False)
    asp = _extract_sub(loaded_state, "aspect_queries")
    if asp:
        model.aspect_queries.load_state_dict(asp, strict=False)
    txt_r = _extract_sub(loaded_state, "text_retriever")
    if txt_r:
        model.text_retriever.load_state_dict(txt_r, strict=False)
    img_r = _extract_sub(loaded_state, "img_retriever")
    if img_r:
        model.img_retriever.load_state_dict(img_r, strict=False)
    roi_r = _extract_sub(loaded_state, "roi_retriever")
    if roi_r:
        model.roi_retriever.load_state_dict(roi_r, strict=False)
    fuse = _extract_sub(loaded_state, "gated_fusion")
    if fuse:
        model.gated_fusion.load_state_dict(fuse, strict=False)
    img_sum = _extract_sub(loaded_state, "img_sum_projector")
    if img_sum:
        model.img_sum_projector.load_state_dict(img_sum, strict=False)


def train(args, tokenizer, device: torch.device, output_dir: Path):
    """Train single model for all 6 aspects."""
    print(f"\n{'='*60}")
    print(f"Training MultimodalACSAModel (all {NUM_ASPECTS} aspects, encode-once, aspect-loop)")
    print(f"{'='*60}")

    train_dataset = MultimodalSentimentDataset(
        split="train",
        data_dir=args.data_dir,
    )
    dev_dataset = MultimodalSentimentDataset(
        split="dev",
        data_dir=args.data_dir,
    )
    test_dataset = MultimodalSentimentDataset(
        split="test",
        data_dir=args.data_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    print("Initializing model (downloading base models if needed)...")
    model = MultimodalACSAModel(use_lora=args.use_lora)
    model.to(device)

    params = model.get_trainable_params()
    print(f"Trainable params: {params}")

    optimizer = build_optimizer(
        model,
        lr_lora=args.lr_lora,
        lr_other=args.lr_other,
        weight_decay=args.weight_decay,
    )
    num_training_steps = len(train_loader) // args.gradient_accumulation * args.max_epochs
    scheduler = build_scheduler(optimizer, num_training_steps, args.warmup_ratio)
    scaler = GradScaler()

    start_epoch = 0
    best_f1 = 0.0
    if args.resume:
        resume_ckpt = Path(args.resume)
        if resume_ckpt.is_dir():
            resume_ckpt = resume_ckpt / "last_checkpoint.pt"
        if resume_ckpt.exists():
            print(f"Resuming from checkpoint: {resume_ckpt}")
            load_model_weights(model, str(resume_ckpt), device=str(device))
            state = torch.load(resume_ckpt, map_location=device, weights_only=False)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            start_epoch = state["epoch"] + 1
            best_f1 = state.get("best_f1", 0.0)
            print(f"  Resumed: epoch={start_epoch}, best_f1={best_f1:.4f}")

    patience_counter = 0
    for epoch in range(start_epoch, args.max_epochs):
        print(f"\nEpoch {epoch + 1}/{args.max_epochs}")

        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            tokenizer=tokenizer,
            gradient_accumulation=args.gradient_accumulation,
            device=device,
        )
        print(f"Train Loss: {train_loss:.4f}")

        eval_results = eval_epoch(
            model=model,
            dataloader=dev_loader,
            tokenizer=tokenizer,
            device=device,
        )
        print(f"Dev Loss: {eval_results['loss']:.4f}")
        print(f"Dev F1 (macro): {eval_results['overall_f1_macro']:.4f}  "
              f"Precision: {eval_results['overall_precision']:.4f}  "
              f"Recall: {eval_results['overall_recall']:.4f}  "
              f"Accuracy: {eval_results['overall_accuracy']:.4f}")
        print("Per-aspect metrics (F1 / Precision / Recall / Acc):")
        for aspect_name, metrics in eval_results["per_aspect"].items():
            print(f"  {aspect_name}: F1={metrics['f1_macro']:.4f}  "
                  f"P={metrics['precision']:.4f}  "
                  f"R={metrics['recall']:.4f}  "
                  f"Acc={metrics['accuracy']:.4f}")

        current_f1 = eval_results["overall_f1_macro"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            print(f"*** New best F1: {best_f1:.4f} (P={eval_results['overall_precision']:.4f}, R={eval_results['overall_recall']:.4f}) ***")

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_f1=best_f1,
                path=str(output_dir / "best_checkpoint.pt"),
            )
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.early_stopping_patience}")

        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_f1=best_f1,
            path=str(output_dir / "last_checkpoint.pt"),
        )

    print(f"\nBest overall F1: {best_f1:.4f}")

    # ── Test evaluation with best model ───────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading best model for test evaluation...")
    print("=" * 60)

    # Reload model with best weights
    model = MultimodalACSAModel(use_lora=args.use_lora)
    model.to(device)
    best_ckpt = str(output_dir / "best_checkpoint.pt")
    load_model_weights(model, best_ckpt, device=str(device))
    print(f"Loaded best_checkpoint_model.safetensors")

    test_results = test_epoch(
        model=model,
        dataloader=test_loader,
        tokenizer=tokenizer,
        device=device,
    )

    print(f"\nTest Loss: {test_results['loss']:.4f}")
    print(f"Test F1 (macro): {test_results['overall_f1_macro']:.4f}  "
          f"Precision: {test_results['overall_precision']:.4f}  "
          f"Recall: {test_results['overall_recall']:.4f}  "
          f"Accuracy: {test_results['overall_accuracy']:.4f}")
    print("Per-aspect metrics (F1 / Precision / Recall / Acc):")
    for aspect_name, metrics in test_results["per_aspect"].items():
        print(f"  {aspect_name}: F1={metrics['f1_macro']:.4f}  "
              f"P={metrics['precision']:.4f}  "
              f"R={metrics['recall']:.4f}  "
              f"Acc={metrics['accuracy']:.4f}")

    # Save test_result.json
    test_result_path = output_dir / "test_result.json"
    with open(test_result_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    print(f"\nTest results saved to {test_result_path}")

    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return best_f1


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ensure_base_models_cached()

    tokenizer = load_tokenizer()

    best_f1 = train(
        args=args,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir,
    )

    results = {
        "best_f1": best_f1,
    }
    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTraining complete. Results saved to {results_path}")
    print(f"Best F1 (macro): {best_f1:.4f}")


if __name__ == "__main__":
    main()
