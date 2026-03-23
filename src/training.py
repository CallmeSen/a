import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score

from .config import COMPUTE_DTYPE, LAMBDA_ACD, LAMBDA_ASC


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


def setup_optimizer(sentiment_model, learning_rate, lora_lr, weight_decay):
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
        [{"params": head_params, "lr": learning_rate}, {"params": lora_params, "lr": lora_lr}],
        weight_decay=weight_decay,
    )
    return optimizer, head_params, lora_params


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


def _optimizer_step(model, optimizer, scheduler, head_params: List[torch.nn.Parameter], lora_params: List[torch.nn.Parameter]):
    _sanitize_grads(model)
    if head_params:
        torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
    if lora_params:
        torch.nn.utils.clip_grad_norm_(lora_params, max_norm=0.3)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def train_epoch(model, dataloader, optimizer, scheduler, run_device, head_params, lora_params, gradient_accumulation_steps=1, tokenizer=None):
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
            _optimizer_step(model, optimizer, scheduler, head_params, lora_params)

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "acd": f"{l_acd.item():.4f}", "asc": f"{l_asc.item():.4f}"})

    if valid_micro_steps % gradient_accumulation_steps != 0:
        _optimizer_step(model, optimizer, scheduler, head_params, lora_params)

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
        pred_combined = pred_presence * pred_sentiment

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
