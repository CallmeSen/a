import os
import json
import math
import warnings
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import AutoModel, AutoModelForCausalLM, AutoImageProcessor, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from safetensors.torch import save_file, load_file
from sklearn.metrics import precision_score, recall_score, f1_score

# Use refactored components from src for explicit fusion architecture
from src.vit_transformer import VisionEncoder
from src.projector_layer import MLPProjector
from src.multimodal_sentiment_model import MultimodalSentimentModel

# ============================================
# GLOBAL CONFIGURATION
# ============================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Use float32 to reduce NaN risk with internlm2_5-1_8b + LoRA
COMPUTE_DTYPE = torch.float32

VISION_MODEL_NAME = "microsoft/swinv2-base-patch4-window8-256"
VISION_HIDDEN_SIZE = 1024
LLM_MODEL_NAME = "internlm/internlm2_5-1_8b"
LLM_HIDDEN_SIZE = 2048

IMAGE_SIZE = 256
MAX_TEXT_LENGTH = 256
MAX_IMAGES = 5
TOP_K_IMAGES = 3

DATA_DIR = "datasets"
IMAGE_DIR = os.path.join(DATA_DIR, "image")
OUTPUT_DIR = "output_model"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.safetensors")
TRAINING_INFO_PATH = os.path.join(OUTPUT_DIR, "training_info.pt")

BATCH_SIZE = 8
NUM_WORKERS = 0

LEARNING_RATE = 2e-5
LORA_LR = 1e-6
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 10
WARMUP_PROPORTION = 0.1
GRADIENT_ACCUMULATION_STEPS = 2
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-4

LAMBDA_ACD = 1.0
LAMBDA_ASC = 1.0

ASPECT_LABELS = ["Facilities", "Public_area", "Location", "Food", "Room", "Service"]
ASPECT2ID = {label: idx for idx, label in enumerate(ASPECT_LABELS)}
ID2ASPECT = {idx: label for idx, label in enumerate(ASPECT_LABELS)}
NUM_ASPECTS = len(ASPECT_LABELS)

CLASS_LABELS = ["None", "Negative", "Neutral", "Positive"]
CLASS2ID = {label: idx for idx, label in enumerate(CLASS_LABELS)}
ID2CLASS = {idx: label for idx, label in enumerate(CLASS_LABELS)}
NUM_CLASSES = len(CLASS_LABELS)

_SENTIMENT_TO_CLASS = {"Negative": 1, "Neutral": 2, "Positive": 3}

# mutable globals wired in main()
head_params: List[torch.nn.Parameter] = []
lora_params: List[torch.nn.Parameter] = []

warnings.filterwarnings("ignore", message=".*AttentionMaskConverter.*")
logging.getLogger("transformers").setLevel(logging.ERROR)


# ============================================
# DATA / TRANSFORMS
# ============================================

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

swin_image_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_NAME)


def _ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def build_transform(input_size: int = IMAGE_SIZE):
    def _transform(img: Image.Image) -> torch.Tensor:
        img = _ensure_rgb(img)
        proc_inputs = swin_image_processor(
            images=img,
            return_tensors="pt",
            do_resize=True,
            size={"height": input_size, "width": input_size},
            do_center_crop=False,
        )
        return proc_inputs["pixel_values"].squeeze(0)

    return _transform


def build_train_transform(input_size: int = IMAGE_SIZE):
    train_aug = T.Compose(
        [
            T.RandomResizedCrop((input_size, input_size), scale=(0.85, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]
    )

    def _transform(img: Image.Image) -> torch.Tensor:
        img = _ensure_rgb(img)
        img = train_aug(img)
        proc_inputs = swin_image_processor(
            images=img,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        return proc_inputs["pixel_values"].squeeze(0)

    return _transform


def load_dataset_json(json_path: str) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {json_path}")
    return data


def load_all_splits(data_dir: str = DATA_DIR) -> Dict[str, list]:
    splits = {}
    for split in ["train", "dev", "test"]:
        json_path = os.path.join(data_dir, f"{split}.json")
        if os.path.exists(json_path):
            splits[split] = load_dataset_json(json_path)
        else:
            print(f"[WARNING] {json_path} not found!")
            splits[split] = []
    return splits


class SentimentDataset(Dataset):
    def __init__(self, data: list, image_dir: str, aspect2id: dict, transform=None):
        self.data = data
        self.image_dir = image_dir
        self.aspect2id = aspect2id
        self.num_aspects = len(aspect2id)
        self.transform = transform if transform else build_transform(IMAGE_SIZE)

        self.samples = self._prepare_samples()
        print(f"Prepared {len(self.samples)} valid samples")

    def _parse_label(self, label: str):
        if "#" not in label:
            return None
        parts = label.split("#")
        if len(parts) != 2:
            return None
        aspect, sentiment = parts[0], parts[1]
        if aspect in self.aspect2id and sentiment in _SENTIMENT_TO_CLASS:
            return (aspect, sentiment)
        return None

    def _prepare_samples(self) -> list:
        valid_samples = []
        for item in self.data:
            if not item.get("list_img") or len(item["list_img"]) == 0:
                continue

            raw_labels = item.get("text_img_label", [])
            if not raw_labels:
                continue

            parsed_labels = []
            for label in raw_labels:
                parsed = self._parse_label(label)
                if parsed is not None:
                    parsed_labels.append(parsed)

            if not parsed_labels:
                continue

            valid_img_paths = []
            for img_name in item["list_img"][:MAX_IMAGES]:
                img_path = os.path.join(self.image_dir, img_name)
                if os.path.exists(img_path):
                    valid_img_paths.append(img_path)

            if not valid_img_paths:
                continue

            valid_samples.append(
                {
                    "comment": item.get("comment", ""),
                    "image_paths": valid_img_paths,
                    "parsed_labels": parsed_labels,
                    "raw_labels": raw_labels,
                }
            )

        return valid_samples

    def __len__(self):
        return len(self.samples)

    def _labels_to_tensor(self, parsed_labels: list) -> torch.Tensor:
        labels = torch.zeros(self.num_aspects, dtype=torch.long)
        for aspect, sentiment in parsed_labels:
            a_id = self.aspect2id[aspect]
            labels[a_id] = _SENTIMENT_TO_CLASS[sentiment]
        return labels

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        image_tensors = []
        for img_path in sample["image_paths"]:
            image = Image.open(img_path).convert("RGB")
            image_tensors.append(self.transform(image))

        pixel_values = torch.stack(image_tensors)
        labels = self._labels_to_tensor(sample["parsed_labels"])

        return {
            "pixel_values": pixel_values,
            "num_images": len(image_tensors),
            "labels": labels,
            "comment": sample["comment"],
            "image_paths": sample["image_paths"],
        }


def make_collate_fn(tokenizer_ref):
    def collate_fn(batch):
        image_counts = [item["num_images"] for item in batch]
        max_imgs = max(image_counts)

        padded_pixels = []
        for item in batch:
            pvs = item["pixel_values"]
            n = pvs.shape[0]
            if n < max_imgs:
                pad = torch.zeros(max_imgs - n, *pvs.shape[1:], dtype=pvs.dtype)
                padded_pixels.append(torch.cat([pvs, pad], dim=0))
            else:
                padded_pixels.append(pvs)

        pixel_values = torch.stack(padded_pixels)
        image_counts_tensor = torch.tensor(image_counts)

        labels = torch.stack([item["labels"] for item in batch])
        comments = [item["comment"] for item in batch]
        image_paths = [item["image_paths"] for item in batch]

        text_inputs = tokenizer_ref(
            comments,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "image_counts": image_counts_tensor,
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
            "labels": labels,
            "comments": comments,
            "image_paths": image_paths,
        }

    return collate_fn


# ============================================
# MODEL COMPONENTS
# ============================================


"""
Local model/encoder/projector implementations removed in favor of refactored
versions under `src/`. See `src/vit_transformer.py`, `src/projector_layer.py`
and `src/multimodal_sentiment_model.py` for the canonical implementations.
"""


# ============================================
# TRAIN / VALIDATE
# ============================================


class LazyLambdaScheduler:
    def __init__(self, optimizer, lr_lambda):
        self.optimizer = optimizer
        self.lr_lambda = lr_lambda
        self.base_lrs = []
        for group in optimizer.param_groups:
            base_lr = group["lr"]
            group.setdefault("initial_lr", base_lr)
            self.base_lrs.append(base_lr)
        self._last_lr = list(self.base_lrs)
        self.last_epoch = -1
        self.started = False

    def _apply(self, step_idx):
        scale = float(self.lr_lambda(step_idx))
        self._last_lr = []
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            lr = base_lr * scale
            group["lr"] = lr
            self._last_lr.append(lr)
        self.last_epoch = step_idx

    def start(self):
        if self.started:
            return
        self.started = True
        self._apply(0)

    def step(self):
        if not self.started:
            self.start()
            return
        self._apply(self.last_epoch + 1)

    def get_last_lr(self):
        return list(self._last_lr)


def compute_loss(outputs, labels, run_device):
    acd_logits = outputs["acd_logits"]
    asc_logits = outputs["asc_logits"]

    if not torch.isfinite(acd_logits).all() or not torch.isfinite(asc_logits).all():
        nan_loss = torch.tensor(float("nan"), device=run_device, dtype=torch.float32)
        return nan_loss, nan_loss, nan_loss

    acd_labels = (labels > 0).float()
    l_acd = F.binary_cross_entropy_with_logits(acd_logits.float(), acd_labels, reduction="mean")

    relevant_mask = labels > 0
    if relevant_mask.any():
        asc_logits_flat = asc_logits[relevant_mask]
        asc_labels_flat = (labels[relevant_mask] - 1).long()
        l_asc = F.cross_entropy(asc_logits_flat.float(), asc_labels_flat, reduction="mean")
    else:
        l_asc = torch.tensor(0.0, device=run_device, dtype=torch.float32)

    total_loss = LAMBDA_ACD * l_acd + LAMBDA_ASC * l_asc
    return total_loss, l_acd.detach(), l_asc.detach()


def _sanitize_grads(model):
    count = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            bad = torch.isnan(p.grad) | torch.isinf(p.grad)
            if bad.any():
                pct = bad.float().mean().item() * 100
                print(f"[WARN-GRAD] {name}: {pct:.1f}% NaN/Inf grads -> zeroed")
                p.grad[bad] = 0.0
                count += 1
    return count


def _optimizer_step(model, optimizer, scheduler):
    n_fixed = _sanitize_grads(model)
    if head_params:
        torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
    if lora_params:
        torch.nn.utils.clip_grad_norm_(lora_params, max_norm=0.3)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return n_fixed


def train_epoch(model, dataloader, optimizer, scheduler, run_device, gradient_accumulation_steps=1, tokenizer=None):
    model.train()
    total_loss = 0.0
    total_acd_loss = 0.0
    total_asc_loss = 0.0
    num_batches = 0
    nan_batches = 0

    optimizer.zero_grad(set_to_none=True)
    scheduler.start()

    progress_bar = tqdm(dataloader, desc="Training")

    valid_micro_steps = 0
    for step, batch in enumerate(progress_bar):
        pixel_values = batch["pixel_values"].to(run_device, dtype=COMPUTE_DTYPE)
        image_counts = batch["image_counts"].to(run_device)
        input_ids = batch["input_ids"].to(run_device)
        attention_mask = batch["attention_mask"].to(run_device)
        labels = batch["labels"].to(run_device)

        outputs = model(pixel_values, input_ids, attention_mask, image_counts)
        loss, l_acd, l_asc = compute_loss(outputs, labels, run_device)

        if step == 0 and tokenizer is not None:
            print(f"[DIAG] tokenizer.padding_side={tokenizer.padding_side}, pad_token_id={tokenizer.pad_token_id}")

        if not torch.isfinite(loss):
            nan_batches += 1
            continue

        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()
        valid_micro_steps += 1

        total_loss += loss.item()
        total_acd_loss += l_acd.item()
        total_asc_loss += l_asc.item()
        num_batches += 1

        if valid_micro_steps % gradient_accumulation_steps == 0:
            _optimizer_step(model, optimizer, scheduler)

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "acd": f"{l_acd.item():.4f}", "asc": f"{l_asc.item():.4f}"})

    if valid_micro_steps % gradient_accumulation_steps != 0:
        _optimizer_step(model, optimizer, scheduler)

    if nan_batches > 0:
        print(f"[WARN] {nan_batches} batches skipped (non-finite loss)")

    if num_batches == 0:
        return float("nan"), float("nan"), float("nan")

    return total_loss / num_batches, total_acd_loss / num_batches, total_asc_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, run_device):
    model.eval()
    total_loss = 0.0
    total_acd_loss = 0.0
    total_asc_loss = 0.0
    num_batches = 0

    all_true_labels = []
    all_pred_labels = []

    for batch in tqdm(dataloader, desc="Validating"):
        pixel_values = batch["pixel_values"].to(run_device, dtype=COMPUTE_DTYPE)
        image_counts = batch["image_counts"].to(run_device)
        input_ids = batch["input_ids"].to(run_device)
        attention_mask = batch["attention_mask"].to(run_device)
        labels = batch["labels"].to(run_device)

        outputs = model(pixel_values, input_ids, attention_mask, image_counts)
        loss, l_acd, l_asc = compute_loss(outputs, labels, run_device)
        if not torch.isfinite(loss):
            continue

        total_loss += loss.item()
        total_acd_loss += l_acd.item()
        total_asc_loss += l_asc.item()
        num_batches += 1

        acd_logits = outputs["acd_logits"]
        asc_logits = outputs["asc_logits"]
        pred_presence = (acd_logits > 0).long()
        pred_sentiment = asc_logits.argmax(dim=-1) + 1
        pred_combined = pred_presence * pred_sentiment  # fixed

        all_true_labels.append(labels.cpu())
        all_pred_labels.append(pred_combined.cpu())

    if len(all_true_labels) == 0:
        print("[WARN] No valid validation batches")
        return float("nan"), float("nan"), float("nan"), 0.0, None, None

    avg_loss = total_loss / num_batches
    avg_acd = total_acd_loss / num_batches
    avg_asc = total_asc_loss / num_batches

    all_true = torch.cat(all_true_labels, dim=0)
    all_pred = torch.cat(all_pred_labels, dim=0)

    true_flat = all_true.numpy().reshape(-1)
    pred_flat = all_pred.numpy().reshape(-1)
    macro_f1 = f1_score(true_flat, pred_flat, average="macro", zero_division=0)

    return avg_loss, avg_acd, avg_asc, macro_f1, all_pred, all_true


# ============================================
# INFERENCE HELPERS
# ============================================


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


def predict_aspect_sentiment(image_path: str, comment: str, model: MultimodalSentimentModel, tokenizer) -> dict:
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


# ============================================
# MAIN
# ============================================


def main():
    global head_params, lora_params

    print(f"Compute dtype: {COMPUTE_DTYPE}")
    print(f"Vision model: {VISION_MODEL_NAME}")
    print(f"LLM model:    {LLM_MODEL_NAME}")
    print(f"Data dir:     {DATA_DIR}")

    # 1) Load dataset
    dataset_splits = load_all_splits(DATA_DIR)

    # 2) Build vision + projector
    vision_encoder = VisionEncoder(model_name=VISION_MODEL_NAME, device=device, torch_dtype=COMPUTE_DTYPE)
    projector = MLPProjector(vision_dim=VISION_HIDDEN_SIZE, llm_dim=LLM_HIDDEN_SIZE).to(device)

    # 3) Load tokenizer + LLM
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"

    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=COMPUTE_DTYPE,
        attn_implementation="eager",
    ).to(device).eval()

    # 4) Datasets / loaders
    train_dataset = SentimentDataset(dataset_splits["train"], IMAGE_DIR, ASPECT2ID, transform=build_train_transform(IMAGE_SIZE))
    dev_dataset = SentimentDataset(dataset_splits["dev"], IMAGE_DIR, ASPECT2ID, transform=build_transform(IMAGE_SIZE))
    test_dataset = SentimentDataset(dataset_splits["test"], IMAGE_DIR, ASPECT2ID, transform=build_transform(IMAGE_SIZE))

    collate_fn = make_collate_fn(tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    # 5) Model — use refactored implementation from src/ (no fallback)
    print("[INFO] Instantiating MultimodalSentimentModel from src/")
    sentiment_model = MultimodalSentimentModel(
        vision_encoder=vision_encoder,
        projector=projector,
        llm_model=llm_model,
        tokenizer=tokenizer,
        num_aspects=NUM_ASPECTS,
        num_sentiment_classes=3,
        top_k_images=TOP_K_IMAGES,
        lora_r=16,
        lora_alpha=16,
        lora_dropout=0.0,
    ).to(device)

    total_params, trainable_params = sentiment_model.get_trainable_params()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 6) Optimizer / scheduler
    sentiment_model.llm.gradient_checkpointing_disable()

    head_named_params = []
    lora_named_params = []
    for name, param in sentiment_model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_named_params.append((name, param))
        else:
            head_named_params.append((name, param))

    head_params = [param for _, param in head_named_params]
    lora_params = [param for _, param in lora_named_params]

    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": LEARNING_RATE}, {"params": lora_params, "lr": LORA_LR}],
        weight_decay=WEIGHT_DECAY,
    )

    steps_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_PROPORTION)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LazyLambdaScheduler(optimizer, lr_lambda)

    # 7) Pre-train sanity
    sentiment_model.eval()
    batch0 = next(iter(train_loader))
    with torch.no_grad():
        pixel_values = batch0["pixel_values"].to(device, dtype=COMPUTE_DTYPE)
        image_counts = batch0["image_counts"].to(device)
        input_ids = batch0["input_ids"].to(device)
        attention_mask = batch0["attention_mask"].to(device)
        out0 = sentiment_model(pixel_values, input_ids, attention_mask, image_counts)
        print(f"Sanity acd_logits finite={torch.isfinite(out0['acd_logits']).all().item()}")
        print(f"Sanity asc_logits finite={torch.isfinite(out0['asc_logits']).all().item()}")
    sentiment_model.train()

    # 8) Training
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_acd, train_asc = train_epoch(
            sentiment_model,
            train_loader,
            optimizer,
            scheduler,
            device,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            tokenizer=tokenizer,
        )
        train_losses.append(train_loss)

        val_loss, val_acd, val_asc, macro_f1, val_preds, _ = validate(sentiment_model, dev_loader, device)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f} (ACD={train_acd:.4f}, ASC={train_asc:.4f})")
        print(f"Val Loss:   {val_loss:.4f} (ACD={val_acd:.4f}, ASC={val_asc:.4f})")
        print(f"Val Macro-F1: {macro_f1:.4f}")

        if not math.isfinite(val_loss) or val_preds is None:
            epochs_without_improvement += 1
        elif macro_f1 > best_macro_f1 + EARLY_STOPPING_MIN_DELTA:
            best_macro_f1 = macro_f1
            epochs_without_improvement = 0

            trainable_param_names = {name for name, p in sentiment_model.named_parameters() if p.requires_grad}
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
                    "num_aspects": NUM_ASPECTS,
                    "num_sentiment_classes": 3,
                    "aspect_labels": ASPECT_LABELS,
                    "class_labels": CLASS_LABELS,
                    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                    "trainable_param_names": sorted(trainable_param_names),
                },
                TRAINING_INFO_PATH,
                _use_new_zipfile_serialization=True,
            )
            print(f"[SAVED] best model with macro-F1={macro_f1:.4f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("[EARLY STOP] triggered")
            break

    # 9) Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses, "b-o", label="Train Loss", linewidth=2)
    ax.plot(epochs_range, val_losses, "r-o", label="Dev Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Dev Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # 10) Evaluate on test
    if os.path.exists(BEST_MODEL_PATH):
        trainable_state = load_file(BEST_MODEL_PATH)
        trainable_state = {k: v.to(device) for k, v in trainable_state.items()}
        sentiment_model.load_state_dict(trainable_state, strict=False)

    test_loss, test_pres, test_sent, test_macro_f1, test_preds, test_labels = validate(sentiment_model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} (presence={test_pres:.4f}, sentiment={test_sent:.4f})")

    if test_preds is not None:
        y_true = test_labels.numpy().ravel()
        y_pred = test_preds.numpy().ravel()
        macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall:    {macro_recall:.4f}")
        print(f"Macro F1:        {macro_f1:.4f}")
        print(f"Dev Macro-F1:    {test_macro_f1:.4f}")

    # 11) Demo inference
    if len(test_dataset.samples) > 0:
        demo_sample = test_dataset.samples[0]
        demo_image_path = demo_sample["image_paths"][0]
        demo_comment = demo_sample["comment"]
        demo_true_labels = demo_sample["raw_labels"]

        print(f"Comment: {demo_comment}")
        print(f"True labels: {demo_true_labels}")

        img = Image.open(demo_image_path)
        plt.figure(figsize=(6, 4))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"True: {demo_true_labels}")
        plt.tight_layout()
        plt.show()

        result = predict_aspect_sentiment(demo_image_path, demo_comment, sentiment_model, tokenizer)
        print("Detected Aspects:", result["detected_aspects"])
        print("Aspect-Sentiments:", result["aspect_sentiments"])


if __name__ == "__main__":
    main()
